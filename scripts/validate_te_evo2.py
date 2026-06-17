#!/usr/bin/env python3
"""
Validate Fp8TELinear on MPS against PyTorch's NATIVE float8_e4m3fn (CPU) as
ground truth, using real Evo 2 checkpoint scales (the TE / bf16+_extra_state
format).

For a TE layer the true forward GEMM is:
    y = dequant( e4m3(x · act_scale) @ e4m3(W · w_scale)ᵀ )
The reference uses torch.float8_e4m3fn (CPU) for the e4m3 step; our emulation
uses quantize_e4m3 on MPS. We diff the layer outputs per layer. This proves the
emulation reproduces the FP8 GEMM, independent of whether a full Evo 2 model
runs end-to-end on MPS.

    python scripts/validate_te_evo2.py [--model evo2_1b_base]
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp8_mps_emulator import Fp8TELinear, quantize_e4m3

FP8_MAX = 448.0


def native_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Ground-truth e4m3 rounding via PyTorch's native cast (CPU), saturating."""
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).to(torch.float32)


def reference_te_forward(x, W, act_scale, w_scale):
    xq = native_e4m3(x.float() * act_scale)
    wq = native_e4m3(W.float() * w_scale)
    return F.linear(xq, wq) * (1.0 / (act_scale * w_scale))


def find_ckpt(model):
    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/**/{model}.pt"), recursive=True)
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_1b_base")
    args = ap.parse_args()

    ckpt = find_ckpt(args.model)
    if not ckpt:
        print(f"checkpoint for {args.model} not in HF cache; skip")
        return 0

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"model: {args.model}   emulation device: {dev}\n")

    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "module" in sd:
        sd = sd["module"]

    # Collect (weight, act_scale, w_scale) for a sample of FP8 layers.
    layers = []
    for k in sd:
        if not k.endswith("._extra_state"):
            continue
        path = k[: -len("._extra_state")]
        w = sd.get(path + ".weight")
        if w is None:
            continue
        v = sd[k]
        try:
            v.seek(0); meta = torch.load(v, map_location="cpu", weights_only=False)
        except Exception:
            continue
        sf = meta.get("scale_fwd") if hasattr(meta, "get") else None
        if sf is None or len(sf) < 2:
            continue
        layers.append((path, w, float(sf[0]), float(sf[1])))

    if not layers:
        print("no FP8 layers found"); return 0
    sample = layers[:3] + layers[len(layers)//2:len(layers)//2+2] + layers[-2:]

    torch.manual_seed(0)
    worst = 0.0
    for path, W, act_s, w_s in sample:
        out_f, in_f = W.shape
        x = torch.randn(1, 4, in_f, dtype=torch.bfloat16)
        ref = reference_te_forward(x, W, act_s, w_s)
        lin = Fp8TELinear(W.data, None, act_s, w_s).to(dev)
        emu = lin(x.to(dev)).float().cpu()
        rel = ((emu - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)).item()
        worst = max(worst, rel)
        print(f"  {path[:46]:46} {str((out_f,in_f)):>14}  rel-err {rel:.2e}")

    print(f"\nworst relative error across {len(sample)} layers: {worst:.2e}")
    ok = worst < 5e-3
    print("PASS — MPS Fp8TELinear matches native e4m3 GEMM" if ok
          else "FAIL — divergence larger than expected")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

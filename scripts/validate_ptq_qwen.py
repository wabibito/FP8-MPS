#!/usr/bin/env python3
"""
Validate Fp8PTQLinear on MPS against PyTorch's NATIVE float8_e4m3fn (CPU) as
ground truth, using real Qwen3-0.6B-FP8 weights.

torch.float8_e4m3fn casts on CPU (PyTorch's reference FP8 path — numerically the
same arithmetic an H100 runs), but raises on MPS. So for each FP8 layer we:
  1. dequantize the stored e4m3 weight via the NATIVE cast on CPU  -> reference
  2. dequantize via our Fp8PTQLinear on MPS                        -> emulated
and diff the layer outputs on a shared random input. Per-layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp8_mps import Fp8PTQLinear

BLOCK = 128


def native_dequant_cpu(w_fp8_bytes: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """Ground truth: interpret the stored bytes AS native e4m3 and dequantize,
    using PyTorch's own float8 support on CPU. Block-scaled."""
    # w_fp8_bytes already loaded as torch.float8_e4m3fn (safetensors preserves it).
    w = w_fp8_bytes.to(torch.float32)  # native FP8 -> fp32 (reference decode)
    out_f, in_f = w.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(BLOCK, 0)[:out_f].repeat_interleave(BLOCK, 1)[:, :in_f]
    return w * s


def main() -> int:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Qwen/Qwen3-0.6B-FP8", "model.safetensors")
    mps = torch.backends.mps.is_available()
    dev = "mps" if mps else "cpu"
    print(f"device for emulation: {dev}\n")

    torch.manual_seed(0)
    results = []
    with safe_open(path, framework="pt") as f:
        keys = [k for k in f.keys() if k.endswith(".weight")
                and (k[:-7] + ".weight_scale_inv") in f.keys()]
        # sample a handful of layers across the model
        sample = keys[:3] + keys[len(keys)//2: len(keys)//2 + 2] + keys[-2:]
        for k in sample:
            w_fp8 = f.get_tensor(k)                       # torch.float8_e4m3fn
            scale_inv = f.get_tensor(k[:-7] + ".weight_scale_inv")
            out_f, in_f = w_fp8.shape
            x = torch.randn(1, 4, in_f, dtype=torch.bfloat16)

            # Reference: native FP8 decode on CPU
            w_ref = native_dequant_cpu(w_fp8, scale_inv)
            ref = F.linear(x.to(torch.float32), w_ref)

            # Emulated: our Fp8PTQLinear on MPS (or CPU fallback)
            lin = Fp8PTQLinear(w_fp8, scale_inv, None, block=BLOCK).to(dev)
            emu = lin(x.to(dev)).float().cpu()

            denom = ref.abs().mean().clamp_min(1e-6)
            rel = (emu - ref).abs().mean() / denom
            results.append((k, out_f, in_f, rel.item()))
            print(f"  {k[:52]:52} {str((out_f,in_f)):>14}  rel-err {rel.item():.2e}")

    worst = max(r[3] for r in results)
    print(f"\nworst relative error across {len(results)} layers: {worst:.2e}")
    # FP8 weights are identical bytes; our dequant should match native to ~bf16 eps.
    ok = worst < 5e-3
    print("PASS — MPS emulation matches native FP8 decode" if ok
          else "FAIL — divergence larger than expected, investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

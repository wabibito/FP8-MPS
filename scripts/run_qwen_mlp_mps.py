#!/usr/bin/env python3
"""
End-to-end demo: run a real FP8 model's compute on MPS via fp8_mps.

Loads the actual FP8 weights from Qwen3-0.6B-FP8, builds the gated-MLP of one
transformer block (gate/up/down projections — the SwiGLU evo2 and Qwen both
use), swaps the three F8_E4M3 linears for Fp8PTQLinear via apply_ptq_emulation,
and runs a forward on MPS. PyTorch/MPS cannot run these F8_E4M3 weights
natively; this shows the library can. Validates the MPS output against the
CPU-native-FP8 reference.

    python scripts/run_qwen_mlp_mps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp8_mps import apply_ptq_emulation

BLOCK = 128


class Fp8Linear(nn.Module):
    """A bare nn.Module holding a stored FP8 weight + scale_inv, so
    apply_ptq_emulation can detect and replace it (mimics how an HF FP8 layer
    looks before our swap)."""
    def __init__(self, w_fp8, scale_inv):
        super().__init__()
        self.weight = nn.Parameter(w_fp8, requires_grad=False)
        self.register_buffer("weight_scale_inv", scale_inv)

    def forward(self, x):  # not used until swapped, but keep it valid
        raise RuntimeError("raw FP8 layer not runnable; call apply_ptq_emulation first")


class GatedMLP(nn.Module):
    def __init__(self, gate, up, down):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = gate, up, down

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def native_dequant(w_fp8, scale_inv):
    """CPU reference: native e4m3 decode, block-scaled."""
    w = w_fp8.to(torch.float32)
    out_f, in_f = w.shape
    s = scale_inv.to(torch.float32)
    s = s.repeat_interleave(BLOCK, 0)[:out_f].repeat_interleave(BLOCK, 1)[:, :in_f]
    return w * s


def main() -> int:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Qwen/Qwen3-0.6B-FP8", "model.safetensors")
    layer = "model.layers.0.mlp"
    parts = {}
    with safe_open(path, framework="pt") as f:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w = f.get_tensor(f"{layer}.{proj}.weight")
            s = f.get_tensor(f"{layer}.{proj}.weight_scale_inv")
            parts[proj] = (w, s)
    in_dim = parts["gate_proj"][0].shape[1]
    print(f"loaded Qwen3-0.6B-FP8 block-0 MLP: hidden={in_dim}, "
          f"dtype={parts['gate_proj'][0].dtype}")

    torch.manual_seed(0)
    x = torch.randn(1, 6, in_dim, dtype=torch.bfloat16)

    # Reference: native FP8 decode on CPU, run the gated MLP in fp32.
    def ref_linear(t, wp):
        w, s = parts[wp]
        return F.linear(t.float(), native_dequant(w, s))
    ref = ref_linear(F.silu(ref_linear(x, "gate_proj")) * ref_linear(x, "up_proj"), "down_proj")

    # Build the MLP from raw FP8 layers, then swap to emulated and run on MPS.
    def mk(wp):
        w, s = parts[wp]
        return Fp8Linear(w, s)
    mlp = GatedMLP(mk("gate_proj"), mk("up_proj"), mk("down_proj"))
    n = apply_ptq_emulation(mlp, block=BLOCK)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    mlp = mlp.to(dev)
    print(f"swapped {n} FP8 linears -> Fp8PTQLinear; running on {dev}")
    with torch.no_grad():
        out = mlp(x.to(dev)).float().cpu()

    rel = ((out - ref).abs().mean() / ref.abs().mean().clamp_min(1e-9)).item()
    print(f"\nMLP output shape: {tuple(out.shape)}  finite: {bool(out.isfinite().all())}")
    print(f"rel error vs CPU-native-FP8 reference: {rel:.3e}")
    ok = n == 3 and out.isfinite().all() and rel < 5e-3
    print("\nPASS — real FP8 model compute runs on MPS, matches native FP8"
          if ok else "FAIL — investigate")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

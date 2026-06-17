"""
Bit-exact FP8 (e4m3 / e5m2) quantization that runs on Apple Silicon (MPS).

PyTorch's MPS backend has no native float8 dtype: ``x.to(torch.float8_e4m3fn)``
works on CPU/CUDA but raises on MPS. These pure-tensor emulations reproduce the
exact rounding of the native cast using only ops MPS supports (log2/round/exp2/
clamp/where), so FP8 models can run on a Mac GPU.

``quantize_e4m3`` is verified bit-exact against ``torch.float8_e4m3fn`` for all
in-range values (saturating, rather than NaN, above the max — which is what FP8
GEMM paths expect after their pre-scaling clamp).
"""

from __future__ import annotations

import torch

# e4m3fn: 4 exp bits, 3 mantissa bits, bias 7, max normal 448, no inf.
FP8_E4M3_MAX = 448.0
_E4M3_MIN_EXP = -6
_E4M3_MANTISSA_BITS = 3

# e5m2: 5 exp bits, 2 mantissa bits, bias 15, max normal 57344.
FP8_E5M2_MAX = 57344.0
_E5M2_MIN_EXP = -14
_E5M2_MANTISSA_BITS = 2


def _round_to_grid(x: torch.Tensor, min_exp: int, mantissa_bits: int,
                   max_val: float) -> torch.Tensor:
    """Round magnitudes to a minifloat grid (round-to-nearest within each binade,
    saturate at max_val, flush sub-subnormals to zero)."""
    orig_dtype = x.dtype
    xf = x.float()
    sign = torch.sign(xf)
    ax = xf.abs()

    e = torch.floor(torch.log2(ax.clamp_min(1e-30)))
    e = torch.clamp(e, min=min_exp)
    step = torch.exp2(e - mantissa_bits)
    q = torch.round(ax / step) * step
    q = torch.clamp(q, max=max_val)

    smallest_subnormal = 2.0 ** (min_exp - mantissa_bits)
    q = torch.where(ax < smallest_subnormal / 2, torch.zeros_like(q), q)
    return (sign * q).to(orig_dtype)


def quantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to the e4m3fn grid (returned in the input dtype). MPS-safe;
    bit-exact vs ``torch.float8_e4m3fn`` for in-range values."""
    return _round_to_grid(x, _E4M3_MIN_EXP, _E4M3_MANTISSA_BITS, FP8_E4M3_MAX)


def quantize_e5m2(x: torch.Tensor) -> torch.Tensor:
    """Round ``x`` to the e5m2 grid (returned in the input dtype). MPS-safe."""
    return _round_to_grid(x, _E5M2_MIN_EXP, _E5M2_MANTISSA_BITS, FP8_E5M2_MAX)


def quantize(x: torch.Tensor, fmt: str = "e4m3") -> torch.Tensor:
    if fmt == "e4m3":
        return quantize_e4m3(x)
    if fmt == "e5m2":
        return quantize_e5m2(x)
    raise ValueError(f"unknown fp8 format {fmt!r} (expected 'e4m3' or 'e5m2')")

"""FP8 (e4m3 / e5m2) emulation for PyTorch on Apple Silicon (MPS).

PyTorch's MPS backend has no native float8. This package provides bit-exact
FP8 quantization and drop-in linear layers so FP8-trained / FP8-quantized
models can run on a Mac GPU — covering both the Transformer Engine training
format (bf16 weights + per-tensor scales) and the standard post-training
format (pre-quantized e4m3 weights + scale_inv).
"""

from .quant import (
    FP8_E4M3_MAX,
    FP8_E5M2_MAX,
    quantize,
    quantize_e4m3,
    quantize_e5m2,
)
from .linear import Fp8TELinear, Fp8PTQLinear

__version__ = "0.1.0"
__all__ = [
    "quantize", "quantize_e4m3", "quantize_e5m2",
    "FP8_E4M3_MAX", "FP8_E5M2_MAX",
    "Fp8TELinear", "Fp8PTQLinear",
]

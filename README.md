# FP8-MPS

Run **FP8 models on Apple Silicon.** PyTorch's MPS backend has no native
`float8` dtype — `x.to(torch.float8_e4m3fn)` works on CPU/CUDA but raises on
MPS, so FP8-trained or FP8-quantized checkpoints can't run on a Mac GPU. This
package fills that gap with **bit-exact** FP8 emulation in pure PyTorch tensor
ops (`log2`/`round`/`exp2`/`clamp`), so the same model that needs an NVIDIA
Hopper GPU can run — numerically faithfully — on an M-series Mac.

It is *emulation*, not hardware FP8: on M1–M4 there's no speedup (the point is
**correctness** — making FP8 checkpoints usable at all). On M5 (native GPU FP8)
the quantizer is the seam to swap for a real FP8 matmul.

## Why two formats?

FP8 checkpoints ship in two incompatible on-disk conventions; this package
handles both:

| | weights stored as | scale info | use |
|---|---|---|---|
| **Transformer Engine** (`Fp8TELinear`) | bf16 | per-tensor `scale_fwd` in a TE `_extra_state` blob | re-quantize to e4m3 to reproduce the training-time GEMM (e.g. Evo 2's Mac checkpoints) |
| **Post-training** (`Fp8PTQLinear`) | actual e4m3 | `weight_scale_inv` (per-tensor or per-block) | dequantize stored FP8 → bf16, then matmul (e.g. Nemotron, DeepSeek-V3, most HF FP8 models) |

## Install

```bash
pip install -e .          # from a checkout
python tests/test_fp8.py  # 7 tests, all should pass
```

## Use

```python
import torch
from fp8_mps import quantize_e4m3, Fp8TELinear, Fp8PTQLinear

# Bit-exact e4m3 rounding that runs on MPS (torch's native cast does not):
q = quantize_e4m3(torch.randn(4, device="mps"))

# TE format: bf16 weight + per-tensor act/weight scales from the checkpoint:
lin = Fp8TELinear(weight_bf16, bias, act_scale=50.0, weight_scale=1500.0)

# PTQ format: pre-quantized e4m3 weight + scale_inv (per-tensor or per-block):
lin = Fp8PTQLinear(weight_fp8, weight_scale_inv, bias, block=128)
```

`quantize_e4m3` is verified bit-exact against `torch.float8_e4m3fn` across
100k random values (it saturates above 448 rather than producing NaN, matching
what FP8 GEMM paths expect after their pre-scale clamp). `quantize_e5m2` is
provided too.

## Status

Core quantizer and both linear formats are implemented and tested. Adapters
that walk a full model and swap its FP8 layers (per format) are the next step —
the first consumer is the [evo2Mac](https://github.com/wabibito/evo2Mac) port,
whose Transformer-Engine emulation this package generalizes.

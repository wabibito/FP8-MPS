# FP8-MPS: Methods

A companion methods note to the evo2Mac paper
([docs/PAPER.md](https://github.com/wabibito/evo2Mac/blob/main/docs/PAPER.md)).
This document covers the general, model-agnostic FP8-on-Apple-Silicon method;
the paper covers the Evo 2 case study and the 20B negative result.

## Problem

PyTorch's MPS backend has no native `float8` dtype. `x.to(torch.float8_e4m3fn)`
works on CPU and CUDA but raises `TypeError` on MPS (PyTorch #132624). So FP8
checkpoints — a fast-growing class (Nemotron, DeepSeek-V3, Qwen3-FP8, Evo 2) —
cannot run on an Apple GPU without an emulation layer.

## Bit-exact e4m3 (and e5m2) on MPS

`quantize_e4m3` rounds to the e4m3fn grid using only MPS-supported ops:

```
e    = clamp(floor(log2(|x|)), min=-6)   # binade exponent (subnormals share -6)
step = exp2(e - 3)                        # mantissa step, 3 mantissa bits
q    = clamp(round(|x| / step) * step, max=448)
q    = where(|x| < 2^(-9)/2, 0, q)        # flush sub-subnormals to 0
return sign(x) * q
```

It **saturates** above 448 rather than producing NaN (what an FP8 GEMM expects
after its pre-scale clamp). Verified **bitwise identical** to
`torch.float8_e4m3fn` on 100% of 100,000 in-range values. `quantize_e5m2` is the
same scheme with exp-min −14 and 2 mantissa bits.

## Two checkpoint formats, two layers

FP8 checkpoints ship in two incompatible conventions:

| | weights | scale | layer |
|---|---|---|---|
| **TE training** | bf16 | per-tensor `scale_fwd` in `_extra_state` | `Fp8TELinear` — *re-quantize* |
| **Post-training (PTQ)** | actual e4m3 | `weight_scale_inv` (per-tensor / per-block) | `Fp8PTQLinear` — *dequantize* |

**`Fp8TELinear`** replays TE's forward GEMM:
`y = round_e4m3(x·s_a) @ round_e4m3(W·s_w)ᵀ / (s_a·s_w) + b`.
Scaling/casting is done in fp32 and the GEMM accumulates in fp32, matching a
hardware FP8 GEMM. (Note: a *consumer* may need a different scaling precision —
Evo 2's `vortex` feeds these projections in bf16, and its checkpoint scales are
tuned to that path; see the paper's engineering log.)

**`Fp8PTQLinear`** dequantizes the stored e4m3 weight by its `weight_scale_inv`
(per-tensor scalar, or per-block expanded over `block × block` tiles) to bf16,
then runs a normal matmul.

## Model-walking adapters

```python
from fp8_mps import apply_te_emulation, apply_ptq_emulation
n = apply_ptq_emulation(model)         # auto-detect e4m3 weight + weight_scale_inv
n = apply_te_emulation(model, scales)  # scales = {path: {"act":…, "weight":…}}
```

## Validation

- Unit: 9/9 (`tests/test_fp8.py`) — bit-exactness, saturation, both linear
  formats (per-tensor + per-block), both adapters, MPS execution.
- Real model (PTQ): `Qwen3-0.6B-FP8` block-0 SwiGLU MLP runs on MPS at
  **4.7×10⁻³** vs native CPU FP8 (`scripts/run_qwen_mlp_mps.py`).
- Real model (TE): Evo 2 `1b_base`/`20b` per-layer at **1.5–2.2×10⁻³**
  (`scripts/validate_te_evo2.py`).

## Scope

This is emulation, not hardware FP8. On M1–M4 there is no speedup; the value is
*correctness* — running FP8 models that otherwise cannot run on an Apple GPU. On
the M5 (native GPU FP8) `quantize_*` is the seam to swap for a real FP8 matmul;
the per-tensor / per-block scales recovered here are exactly what such a path
consumes.

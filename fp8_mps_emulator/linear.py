"""
FP8-emulated linear layers for MPS, covering the two FP8 checkpoint formats:

1. **TE mode** (``Fp8TELinear``) — weights stored in bf16, per-tensor activation
   and weight scales recovered from a Transformer Engine ``_extra_state`` blob.
   The layer re-quantizes to e4m3 to reproduce TE's training-time GEMM:
       y = (q(x·act_scale) @ q(W·w_scale)ᵀ) / (act_scale·w_scale) + b
   (This is the path Evo 2's bf16 Mac checkpoints need.)

2. **PTQ mode** (``Fp8PTQLinear``) — weights stored as actual e4m3 tensors with a
   ``weight_scale_inv`` (per-tensor or per-block). The layer dequantizes the
   stored FP8 weight to bf16 and runs a normal matmul:
       y = x @ (W_fp8 · weight_scale_inv)ᵀ + b
   (This is the path Nemotron / DeepSeek-V3 / most HF FP8 checkpoints use.)

Both return a bare tensor by default; set ``return_tuple=True`` to mirror a
module (e.g. Transformer Engine's ``Linear``) that returns ``(out, bias)``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import quantize


class Fp8TELinear(nn.Module):
    """bf16 weight + per-tensor scales -> emulate TE's e4m3 forward GEMM."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        act_scale: float,
        weight_scale: float,
        fmt: str = "e4m3",
        return_tuple: bool = False,
    ):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.fmt = fmt
        self.return_tuple = return_tuple
        self.weight = nn.Parameter(weight)
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)
        self.register_buffer("act_scale", torch.tensor(float(act_scale)))
        self.register_buffer("weight_scale", torch.tensor(float(weight_scale)))

    def forward(self, x):
        w = self.weight
        # Scale in fp32 BEFORE quantizing. bf16's 8-bit mantissa rounds
        # `x * act_scale` enough to shift many values to the wrong e4m3 bin
        # (measured: bf16 scaling diverges ~2e-2 from native; fp32 scaling ~1e-6).
        # TE likewise scales/casts in higher precision; only GEMM operands are e4m3.
        x_q = quantize(x.float() * self.act_scale, self.fmt)
        w_q = quantize(w.float() * self.weight_scale, self.fmt)
        # Accumulate in fp32 to match a hardware FP8 GEMM (H100 accumulates in fp32).
        out = F.linear(x_q, w_q) * (1.0 / (self.act_scale * self.weight_scale))
        if self.bias is not None:
            out = out + self.bias.float()
        return (out.to(w.dtype), self.bias) if self.return_tuple else out.to(w.dtype)


class Fp8PTQLinear(nn.Module):
    """Pre-quantized e4m3 weight + scale_inv -> dequantize and matmul.

    ``weight_fp8`` is the stored FP8 weight (any dtype holding e4m3 values).
    ``weight_scale_inv`` is the dequant scale: a scalar (per-tensor) or a 2D
    tensor of per-block scales broadcast over ``block`` x ``block`` tiles.
    """

    def __init__(
        self,
        weight_fp8: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        bias: Optional[torch.Tensor],
        block: int = 128,
        return_tuple: bool = False,
    ):
        super().__init__()
        self.out_features, self.in_features = weight_fp8.shape
        self.block = block
        self.return_tuple = return_tuple
        # Dequantize once at load to bf16 (memory-equivalent to a bf16 layer; the
        # point here is correctness on MPS, not FP8 memory savings).
        w = self._dequantize(weight_fp8.float(), weight_scale_inv.float(), block)
        self.weight = nn.Parameter(w.to(torch.bfloat16))
        if bias is not None:
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter("bias", None)

    @staticmethod
    def _dequantize(w_fp8: torch.Tensor, scale_inv: torch.Tensor, block: int) -> torch.Tensor:
        if scale_inv.ndim == 0 or scale_inv.numel() == 1:
            return w_fp8 * scale_inv  # per-tensor
        # Per-block: scale_inv is (ceil(out/block), ceil(in/block)); expand to
        # full weight shape by repeating each scale over its block tile.
        out_f, in_f = w_fp8.shape
        s = scale_inv
        s = s.repeat_interleave(block, dim=0)[:out_f]
        s = s.repeat_interleave(block, dim=1)[:, :in_f]
        return w_fp8 * s

    def forward(self, x):
        out = F.linear(x.to(self.weight.dtype), self.weight, self.bias)
        return (out, self.bias) if self.return_tuple else out

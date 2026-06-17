"""
Model-walking adapters: swap a model's FP8 layers for emulated ones in place,
so running an FP8 model on MPS is plug-and-play.

Two entry points, one per checkpoint format:

- ``apply_te_emulation(model, scales)`` — TE format. ``scales`` maps
  ``module_path -> {"act": float, "weight": float}`` (recover it from the
  checkpoint's ``_extra_state`` blobs with your own loader). Each named module
  with a ``.weight`` is replaced by an ``Fp8TELinear``.

- ``apply_ptq_emulation(model)`` — PTQ format. Finds ``nn.Linear`` modules whose
  weight dtype is ``torch.float8_e4m3fn`` (or a sibling ``*.weight_scale_inv``
  buffer) and replaces them with ``Fp8PTQLinear`` so they run on MPS.

Both return the number of layers replaced.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .linear import Fp8TELinear, Fp8PTQLinear


def _get_parent(model: nn.Module, path: str):
    """Return (parent_module, attr_name) for a dotted module path, or (None, None)."""
    parent_path, _, attr = path.rpartition(".")
    parent = dict(model.named_modules()).get(parent_path) if parent_path else model
    return parent, attr


def apply_te_emulation(model: nn.Module, scales: Dict[str, Dict[str, float]],
                       fmt: str = "e4m3") -> int:
    """Replace TE-format FP8 linears with Fp8TELinear, using per-tensor scales.

    ``scales[path] = {"act": ..., "weight": ...}``. A module is wrapped only if
    it exists, has a ``.weight``, and has a scale entry. The replacement mirrors
    the original's return convention (tuple if it had ``te_return_bias``).
    """
    replaced = 0
    modules = dict(model.named_modules())
    for path, sc in scales.items():
        parent, attr = _get_parent(model, path)
        if parent is None:
            continue
        old = getattr(parent, attr, None)
        if old is None or getattr(old, "weight", None) is None:
            continue
        new = Fp8TELinear(
            weight=old.weight.data,
            bias=old.bias.data if getattr(old, "bias", None) is not None else None,
            act_scale=sc["act"], weight_scale=sc["weight"], fmt=fmt,
            return_tuple=hasattr(old, "te_return_bias"),
        ).to(old.weight.device)
        setattr(parent, attr, new)
        replaced += 1
    return replaced


def apply_ptq_emulation(model: nn.Module, block: int = 128) -> int:
    """Replace PTQ-format FP8 linears (e4m3 weight + ``weight_scale_inv``) with
    Fp8PTQLinear so they run on MPS. Detects layers by an fp8 weight dtype with a
    sibling ``<name>.weight_scale_inv`` parameter/buffer on the same module."""
    replaced = 0
    fp8_dtypes = {torch.float8_e4m3fn, getattr(torch, "float8_e5m2", None)}
    for path, mod in list(model.named_modules()):
        w = getattr(mod, "weight", None)
        scale_inv = getattr(mod, "weight_scale_inv", None)
        if w is None or scale_inv is None or w.dtype not in fp8_dtypes:
            continue
        parent, attr = _get_parent(model, path)
        if parent is None:
            continue
        bias = getattr(mod, "bias", None)
        new = Fp8PTQLinear(
            weight_fp8=w.data, weight_scale_inv=scale_inv.data,
            bias=bias.data if bias is not None else None, block=block,
        ).to(w.device if w.device.type != "meta" else "cpu")
        setattr(parent, attr, new)
        replaced += 1
    return replaced

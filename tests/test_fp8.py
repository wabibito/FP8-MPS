"""Tests for fp8_mps. Run: python -m pytest tests/ (or python tests/test_fp8.py)."""

import torch
import torch.nn.functional as F

from fp8_mps import quantize_e4m3, quantize_e5m2, Fp8TELinear, Fp8PTQLinear


def test_e4m3_bit_exact_vs_native():
    """Emulated e4m3 matches torch.float8_e4m3fn for all in-range values."""
    torch.manual_seed(0)
    x = torch.randn(100_000) * 3.0
    ref = x.to(torch.float8_e4m3fn).to(torch.float32)
    emu = quantize_e4m3(x)
    mask = ref.isfinite()  # native maps >448 to NaN; we saturate
    assert torch.equal(ref[mask], emu[mask]), "e4m3 emulation diverges from native cast"


def test_e4m3_saturates_not_nan():
    x = torch.tensor([500.0, -1000.0, 448.0])
    q = quantize_e4m3(x)
    assert q.isfinite().all()
    assert q.abs().max().item() <= 448.0


def test_e5m2_roundtrip_sane():
    x = torch.tensor([0.5, 1.0, 2.0, 100.0, -7.0])
    q = quantize_e5m2(x)
    # e5m2 represents powers of two exactly
    assert torch.allclose(q[[0, 1, 2]], x[[0, 1, 2]])


def test_te_linear_matches_native_fp8_gemm():
    """Fp8TELinear matches PyTorch's native float8_e4m3fn GEMM (the H100 path).

    Scaling and casting happen in fp32 (like TE); only the GEMM operands are
    e4m3. Reference uses torch.float8_e4m3fn directly.
    """
    torch.manual_seed(1)
    W = torch.randn(64, 32, dtype=torch.bfloat16) * 0.1
    x = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    act_scale, w_scale = 50.0, 1500.0
    lin = Fp8TELinear(W, None, act_scale, w_scale)
    out = lin(x)

    def nat(t):  # native e4m3 round (saturating), fp32
        return t.clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.float32)
    ref = F.linear(nat(x.float() * act_scale), nat(W.float() * w_scale)) / (act_scale * w_scale)
    # e4m3 has ~2 sig figs; small synthetic GEMMs land within a few e-3 of native.
    rel = (out.float() - ref.float()).abs().mean() / ref.float().abs().mean().clamp_min(1e-6)
    assert rel < 5e-3, f"rel err {rel:.2e} too high vs native FP8"
    assert out.shape == (1, 8, 64)


def test_ptq_linear_per_tensor():
    """Fp8PTQLinear dequantizes a per-tensor scaled FP8 weight correctly."""
    torch.manual_seed(2)
    W_real = torch.randn(64, 32) * 0.05
    scale_inv = torch.tensor(W_real.abs().max().item() / 448.0)
    W_fp8 = quantize_e4m3(W_real / scale_inv)  # stored "fp8" values
    x = torch.randn(1, 8, 32, dtype=torch.bfloat16)
    lin = Fp8PTQLinear(W_fp8, scale_inv, None)
    out = lin(x)
    ref = F.linear(x.to(torch.bfloat16), (W_fp8 * scale_inv).to(torch.bfloat16))
    assert torch.allclose(out.float(), ref.float(), atol=1e-2)


def test_ptq_linear_per_block():
    """Per-block scale_inv expands correctly to the weight shape."""
    torch.manual_seed(3)
    out_f, in_f, block = 256, 128, 128
    W_fp8 = quantize_e4m3(torch.randn(out_f, in_f) * 0.1)
    scale_inv = torch.rand(out_f // block, in_f // block) * 0.01 + 0.001
    lin = Fp8PTQLinear(W_fp8, scale_inv, None, block=block)
    assert lin.weight.shape == (out_f, in_f)
    assert lin.weight.isfinite().all()


def test_apply_te_emulation_walks_model():
    """apply_te_emulation swaps named modules and the model still runs."""
    from fp8_mps import apply_te_emulation, Fp8TELinear
    m = torch.nn.Sequential(torch.nn.Linear(32, 64, bias=False),
                            torch.nn.ReLU(),
                            torch.nn.Linear(64, 16, bias=False))
    scales = {"0": {"act": 50.0, "weight": 1500.0},
              "2": {"act": 40.0, "weight": 1200.0}}
    n = apply_te_emulation(m, scales)
    assert n == 2
    assert isinstance(m[0], Fp8TELinear) and isinstance(m[2], Fp8TELinear)
    out = m(torch.randn(1, 8, 32, dtype=torch.bfloat16))
    assert out.shape == (1, 8, 16) and out.isfinite().all()


def test_apply_ptq_emulation_detects_fp8_layers():
    """apply_ptq_emulation finds fp8 weight + weight_scale_inv layers."""
    from fp8_mps import apply_ptq_emulation, Fp8PTQLinear
    lin = torch.nn.Linear(128, 256, bias=False)
    # masquerade as a stored PTQ fp8 layer
    lin.weight = torch.nn.Parameter(
        quantize_e4m3(lin.weight.data * 0.1).to(torch.float8_e4m3fn), requires_grad=False)
    lin.register_buffer("weight_scale_inv", torch.rand(2, 1) * 0.01 + 0.001)
    m = torch.nn.Sequential(lin, torch.nn.ReLU())
    n = apply_ptq_emulation(m, block=128)
    assert n == 1 and isinstance(m[0], Fp8PTQLinear)
    out = m(torch.randn(1, 4, 128, dtype=torch.bfloat16))
    assert out.shape == (1, 4, 256) and out.isfinite().all()


def test_runs_on_mps():
    if not torch.backends.mps.is_available():
        return
    W = torch.randn(64, 32, dtype=torch.bfloat16) * 0.1
    lin = Fp8TELinear(W, None, 50.0, 1500.0).to("mps")
    out = lin(torch.randn(1, 8, 32, dtype=torch.bfloat16, device="mps"))
    assert out.device.type == "mps" and out.isfinite().all()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    raise SystemExit(0 if passed == len(fns) else 1)

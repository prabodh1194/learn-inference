"""Lecture 19 -- quantization.

The only optimization in this book that can make the model WORSE, so the
quality harness is tested alongside the numerics.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestQuantizeDequantize:
    """Round-trip error bounds -- no GPU needed."""

    def test_int8_roundtrip_is_close(self):
        from kernels.quantize import dequantize, quantize_per_channel

        torch.manual_seed(0)
        w = torch.randn(256, 512)
        q, scale, zp = quantize_per_channel(w, bits=8)
        torch.testing.assert_close(dequantize(q, scale, zp), w, rtol=0.05, atol=0.05)

    def test_per_channel_beats_per_tensor_with_outliers(self):
        """One outlier channel stretches a per-tensor range and crushes
        precision everywhere else. This is why per-channel is standard."""
        from kernels.quantize import dequantize, quantize_per_channel, quantize_per_tensor

        torch.manual_seed(0)
        w = torch.randn(64, 128)
        w[0] *= 100.0                            # one wild channel

        q1, s1, z1 = quantize_per_tensor(w, bits=8)
        q2, s2, z2 = quantize_per_channel(w, bits=8)
        err_tensor = (dequantize(q1, s1, z1) - w)[1:].abs().mean()
        err_channel = (dequantize(q2, s2, z2) - w)[1:].abs().mean()
        assert err_channel < err_tensor

    def test_int4_is_lossier_than_int8(self):
        from kernels.quantize import dequantize, quantize_per_channel

        torch.manual_seed(0)
        w = torch.randn(128, 256)
        errs = []
        for bits in (8, 4):
            q, s, z = quantize_per_channel(w, bits=bits)
            errs.append((dequantize(q, s, z) - w).abs().mean().item())
        assert errs[1] > errs[0], "int4 should have more error than int8"

    def test_memory_scales_with_bits(self):
        from kernels.quantize import quantized_bytes

        fp16 = quantized_bytes(n_params=1_000_000, bits=16, group_size=None)
        int8 = quantized_bytes(n_params=1_000_000, bits=8, group_size=128)
        int4 = quantized_bytes(n_params=1_000_000, bits=4, group_size=128)
        assert int8 < fp16 and int4 < int8
        assert int8 / fp16 == pytest.approx(0.5, abs=0.05), "int8 ~ half of fp16"


class TestQualityHarness:
    """Built BEFORE benchmarking speed -- that ordering is the lecture."""

    def test_grades_a_verifiable_task(self):
        from kernels.quality_eval import grade

        assert grade(prediction="42", answer="42") == 1.0
        assert grade(prediction="41", answer="42") == 0.0

    def test_reports_per_task_not_just_aggregate(self):
        """An aggregate score hides a specific broken capability -- exactly
        what perplexity does."""
        from kernels.quality_eval import evaluate

        res = evaluate(
            predictions={"arith": ["4", "6"], "format": ["{}", "not json"]},
            answers={"arith": ["4", "6"], "format": ["{}", "{}"]},
        )
        assert res["arith"] == 1.0
        assert res["format"] == 0.5
        assert "overall" in res


@pytest.mark.cuda
def test_quant_matmul_matches_fp16():
    from kernels.triton.quant_matmul import quant_matmul

    torch.manual_seed(0)
    x = torch.randn(64, 512, device="cuda", dtype=torch.float16)
    w = torch.randn(512, 256, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(quant_matmul(x, w, bits=8), x @ w,
                               rtol=0.1, atol=0.1)

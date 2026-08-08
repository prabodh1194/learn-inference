"""Lecture 16 -- Triton basics.

Every kernel must match PyTorch numerically BEFORE you look at its speed. A
kernel that is fast and wrong is worthless, and the errors here are subtle --
they surface as slightly worse output quality, not crashes.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cuda
torch = pytest.importorskip("torch")


@pytest.fixture
def x():
    torch.manual_seed(0)
    return torch.randn(128, 1024, device="cuda", dtype=torch.float32)


class TestSoftmax:
    def test_matches_torch(self, x):
        from kernels.triton.softmax import softmax

        torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1),
                                   rtol=1e-4, atol=1e-4)

    def test_non_power_of_two_columns(self):
        """1000 columns with BLOCK_SIZE=1024 -- without masking, the ragged
        edge reads garbage and corrupts the row sum."""
        from kernels.triton.softmax import softmax

        y = torch.randn(8, 1000, device="cuda")
        torch.testing.assert_close(softmax(y), torch.softmax(y, dim=-1),
                                   rtol=1e-4, atol=1e-4)

    def test_numerically_stable_on_large_logits(self):
        """Without subtracting the row max, exp() overflows to inf."""
        from kernels.triton.softmax import softmax

        y = torch.full((4, 256), 100.0, device="cuda")
        y[:, 0] = 200.0
        out = softmax(y)
        assert torch.isfinite(out).all(), "overflow -- subtract the max first"
        torch.testing.assert_close(out.sum(-1), torch.ones(4, device="cuda"),
                                   rtol=1e-4, atol=1e-4)


class TestRMSNorm:
    def test_matches_torch(self, x):
        from kernels.triton.rmsnorm import rmsnorm

        w = torch.randn(1024, device="cuda")
        eps = 1e-6
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
        torch.testing.assert_close(rmsnorm(x, w, eps), ref, rtol=1e-4, atol=1e-4)

    def test_no_mean_subtraction(self):
        """RMSNorm is not LayerNorm -- it does NOT centre the input."""
        from kernels.triton.rmsnorm import rmsnorm

        y = torch.full((2, 128), 3.0, device="cuda")
        w = torch.ones(128, device="cuda")
        out = rmsnorm(y, w, 1e-6)
        assert out.abs().mean() > 0.5, "output centred -- that's LayerNorm"

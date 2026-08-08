"""Lecture 20 -- raw CUDA.

Correctness first, as always. The lecture's real outcome is understanding
what Triton was doing for you, which a test can't assert -- but a kernel that
doesn't match PyTorch teaches nothing.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cuda
torch = pytest.importorskip("torch")


def test_cuda_softmax_matches_torch():
    from kernels.cuda import load_extension

    ext = load_extension()
    x = torch.randn(64, 1024, device="cuda")
    torch.testing.assert_close(ext.softmax(x), torch.softmax(x, dim=-1),
                               rtol=1e-4, atol=1e-4)


def test_reduction_stages_agree():
    """Naive, sequential-addressing, and warp-shuffle must all agree.

    They differ only in memory access pattern -- which is the entire lesson,
    and the reason the speedup between them is large.
    """
    from kernels.cuda import load_extension

    ext = load_extension()
    x = torch.randn(4096, device="cuda")
    expected = x.sum()
    for name in ("reduce_naive", "reduce_sequential", "reduce_shuffle"):
        got = getattr(ext, name)(x)
        torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)

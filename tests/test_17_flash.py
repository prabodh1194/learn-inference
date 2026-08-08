"""Lecture 17 -- FlashAttention.

FlashAttention is EXACT, not approximate -- it computes standard attention in
a different order. So the test is equality with the reference, not similarity.

The classic bug is forgetting to rescale the accumulator when the running max
updates. Attention is a weighted average, so the output still looks plausible
and only a numeric check catches it.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cuda
torch = pytest.importorskip("torch")


def reference_attention(q, k, v, causal=True):
    """Standard attention. Materializes the N x N matrix -- the thing
    FlashAttention avoids and the thing we compare against."""
    scale = q.shape[-1] ** -0.5
    scores = (q @ k.transpose(-2, -1)) * scale
    if causal:
        n = scores.shape[-1]
        mask = torch.triu(torch.ones(n, n, device=q.device, dtype=torch.bool), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


@pytest.mark.parametrize("seq_len", [128, 512, 2048])
def test_matches_reference(seq_len):
    from kernels.triton.flash_attention import flash_attention

    torch.manual_seed(0)
    shape = (1, 8, seq_len, 128)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3))
    torch.testing.assert_close(
        flash_attention(q, k, v, causal=True),
        reference_attention(q, k, v, causal=True),
        rtol=1e-2, atol=1e-2,
    )


def test_causal_mask_does_not_leak_the_future():
    """If masking is wrong at tile boundaries, a query attends to later keys.

    Uniquely confusing: the model looks BETTER at predicting, because it's
    cheating.
    """
    from kernels.triton.flash_attention import flash_attention

    torch.manual_seed(0)
    q, k = (torch.randn(1, 1, 64, 64, device="cuda", dtype=torch.float16) for _ in range(2))
    v = torch.zeros(1, 1, 64, 64, device="cuda", dtype=torch.float16)
    v[0, 0, 32:] = 100.0                     # only later positions are non-zero

    out = flash_attention(q, k, v, causal=True)
    assert out[0, 0, :32].abs().max() < 1.0, (
        "early queries picked up mass from later positions -- causal mask leaks"
    )


def test_memory_is_linear_not_quadratic():
    """The other half of the win: no N x N scratch buffer."""
    from kernels.triton.flash_attention import flash_attention

    def peak_for(n):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        q, k, v = (torch.randn(1, 4, n, 64, device="cuda", dtype=torch.float16)
                   for _ in range(3))
        flash_attention(q, k, v, causal=True)
        return torch.cuda.max_memory_allocated()

    # 4x the sequence length would be 16x memory if it were quadratic
    assert peak_for(2048) < peak_for(512) * 8

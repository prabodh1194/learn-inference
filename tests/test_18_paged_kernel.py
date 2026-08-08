"""Lecture 18 -- paged attention kernel.

Three implementations must agree: PyTorch reference, contiguous flash, and
paged flash. Three-way agreement is strong evidence you got the block-table
indirection right.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cuda
torch = pytest.importorskip("torch")


def make_paged(k, v, block_size=16):
    """Scatter contiguous K/V into shuffled physical blocks + a block table.

    Deliberately shuffled: a kernel that ignores the table and reads
    sequentially would pass on an identity mapping and fail here.
    """
    import random

    n = k.shape[2]
    n_blocks = -(-n // block_size)
    order = list(range(n_blocks))
    random.Random(0).shuffle(order)

    cache_k = torch.zeros(n_blocks, block_size, *k.shape[1:2], k.shape[-1],
                          device=k.device, dtype=k.dtype)
    cache_v = torch.zeros_like(cache_k)
    for logical, physical in enumerate(order):
        lo, hi = logical * block_size, min((logical + 1) * block_size, n)
        cache_k[physical, : hi - lo] = k[0, :, lo:hi].transpose(0, 1)
        cache_v[physical, : hi - lo] = v[0, :, lo:hi].transpose(0, 1)
    return cache_k, cache_v, order


@pytest.mark.parametrize("seq_len", [64, 512, 1000])
def test_paged_matches_contiguous(seq_len):
    from kernels.triton.flash_attention import flash_attention
    from kernels.triton.paged_attention import paged_attention

    torch.manual_seed(0)
    q = torch.randn(1, 8, 1, 128, device="cuda", dtype=torch.float16)  # decode
    k = torch.randn(1, 8, seq_len, 128, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 8, seq_len, 128, device="cuda", dtype=torch.float16)

    cache_k, cache_v, table = make_paged(k, v)
    torch.testing.assert_close(
        paged_attention(q, cache_k, cache_v, [table], [seq_len], block_size=16),
        flash_attention(q, k, v, causal=False),
        rtol=1e-2, atol=1e-2,
    )


def test_partial_last_block_is_masked():
    """37 tokens at block_size 16 leaves 11 unwritten slots in block 3.

    Unmasked, those zeros get softmax weight and silently dilute the output.
    """
    from kernels.triton.flash_attention import flash_attention
    from kernels.triton.paged_attention import paged_attention

    torch.manual_seed(0)
    n = 37
    q = torch.randn(1, 4, 1, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 4, n, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 4, n, 64, device="cuda", dtype=torch.float16)

    cache_k, cache_v, table = make_paged(k, v)
    torch.testing.assert_close(
        paged_attention(q, cache_k, cache_v, [table], [n], block_size=16),
        flash_attention(q, k, v, causal=False),
        rtol=1e-2, atol=1e-2,
    )

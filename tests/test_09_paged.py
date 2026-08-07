"""Lecture 09 -- paged attention.

Block accounting is testable without a GPU, so most of this runs on a laptop.
The rule these enforce: paging changes WHERE the cache lives, never what the
model computes.

The leak tests matter more than they look. A block manager that loses blocks
degrades slowly -- capacity shrinks over hours until the server stops admitting
requests, with no crash and no obvious cause.
"""

from __future__ import annotations

import pytest


def _seq(seq_id: int = 0, n_tokens: int = 40):
    from engine.sequence import Sequence

    return Sequence(seq_id=seq_id, prompt_ids=list(range(n_tokens)))


def _bm(n_blocks: int = 16, block_size: int = 16, **kw):
    from engine.block_manager import BlockManager

    return BlockManager(n_blocks=n_blocks, block_size=block_size, **kw)


class TestBlockMath:
    def test_blocks_needed_rounds_up(self):
        bm = _bm(block_size=16)
        assert bm.blocks_needed(1) == 1
        assert bm.blocks_needed(16) == 1
        assert bm.blocks_needed(17) == 2
        assert bm.blocks_needed(40) == 3

    def test_empty_sequence_needs_no_blocks(self):
        assert _bm(block_size=16).blocks_needed(0) == 0


class TestAllocation:
    def test_allocate_gives_enough_blocks(self):
        bm = _bm(n_blocks=16, block_size=16)
        seq = _seq(n_tokens=40)
        bm.allocate(seq)
        assert len(seq.block_table) == 3

    def test_blocks_are_distinct(self):
        """Two sequences must never be handed the same physical block.

        Without prefix caching this is silent corruption: one sequence
        overwrites another's keys and both produce plausible garbage.
        """
        bm = _bm(n_blocks=16)
        a, b = _seq(0, 40), _seq(1, 40)
        bm.allocate(a)
        bm.allocate(b)
        assert not set(a.block_table) & set(b.block_table)

    def test_can_allocate_reports_honestly(self):
        bm = _bm(n_blocks=2, block_size=16)
        assert bm.can_allocate(_seq(n_tokens=32))       # exactly 2 blocks
        assert not bm.can_allocate(_seq(n_tokens=33))   # needs 3

    def test_allocation_fails_cleanly_when_full(self):
        bm = _bm(n_blocks=2, block_size=16)
        bm.allocate(_seq(0, 32))
        assert not bm.can_allocate(_seq(1, 16))


class TestFree:
    def test_free_returns_blocks_to_the_pool(self):
        bm = _bm(n_blocks=4, block_size=16)
        seq = _seq(n_tokens=64)
        bm.allocate(seq)
        assert not bm.can_allocate(_seq(1, 16))
        bm.free(seq)
        assert bm.can_allocate(_seq(1, 64))

    def test_free_clears_the_block_table(self):
        bm = _bm()
        seq = _seq(n_tokens=40)
        bm.allocate(seq)
        bm.free(seq)
        assert seq.block_table == []

    def test_no_leaks_over_many_cycles(self):
        """Allocate/free 200 times. Capacity must be identical at the end.

        A slow leak here looks like a server that quietly stops accepting
        traffic after a few hours.
        """
        bm = _bm(n_blocks=8, block_size=16)
        for i in range(200):
            seq = _seq(i, n_tokens=(i % 5 + 1) * 16)
            assert bm.can_allocate(seq), f"exhausted after {i} cycles -- leak"
            bm.allocate(seq)
            bm.free(seq)
        big = _seq(999, n_tokens=128)
        assert bm.can_allocate(big), "full capacity not restored"


class TestGrowth:
    def test_append_only_allocates_on_boundary(self):
        """Decode adds one token per step; only 1 step in block_size crosses
        a boundary. Allocating every step would exhaust memory immediately."""
        bm = _bm(n_blocks=8, block_size=16)
        seq = _seq(n_tokens=16)
        bm.allocate(seq)
        assert len(seq.block_table) == 1

        for _ in range(15):
            seq.output_ids.append(1)
            bm.append_token(seq)
        assert len(seq.block_table) == 2, "should have crossed exactly once"

    def test_growth_survives_many_tokens(self):
        bm = _bm(n_blocks=32, block_size=16)
        seq = _seq(n_tokens=8)
        bm.allocate(seq)
        for _ in range(100):
            seq.output_ids.append(1)
            bm.append_token(seq)
        assert len(seq.block_table) == bm.blocks_needed(len(seq))


class TestFragmentationMath:
    """The demo's arithmetic -- no GPU, no BlockManager."""

    def test_paged_beats_contiguous_at_long_context(self):
        from book.code.fragmentation import contiguous_sequences, paged_sequences
        from book.code.roofline import ModelDims

        d = ModelDims()
        budget = 20 * 1024**3
        lengths = [512] * 500
        c = contiguous_sequences(budget, d, max_seq_len=32768)
        p = paged_sequences(budget, d, lengths)
        assert p > c * 5

    def test_paged_capacity_is_flat_in_max_seq_len(self):
        """Contiguous is punished for context it never uses. Paged isn't."""
        from book.code.fragmentation import contiguous_sequences, paged_sequences
        from book.code.roofline import ModelDims

        d = ModelDims()
        budget = 20 * 1024**3
        lengths = [512] * 500
        assert paged_sequences(budget, d, lengths) == paged_sequences(
            budget, d, lengths
        )
        assert contiguous_sequences(budget, d, 4096) > contiguous_sequences(
            budget, d, 32768
        )

    def test_internal_fragmentation_is_bounded_by_block_size(self):
        """The core guarantee: waste per sequence < block_size, always."""
        block_size = 16
        for length in (1, 15, 16, 17, 100, 1000, 4095):
            blocks = -(-length // block_size)
            waste = blocks * block_size - length
            assert 0 <= waste < block_size


@pytest.mark.slow
def test_paged_output_matches_contiguous(model_and_tokenizer):
    """Storage layout must not change what the model says."""
    from engine.generate import generate_cached, generate_paged

    model, tok = model_and_tokenizer
    prompt = "The KV cache exists because"
    assert generate_paged(model, tok, prompt, max_tokens=16) == generate_cached(
        model, tok, prompt, max_tokens=16
    )

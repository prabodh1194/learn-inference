"""Lecture 10 -- prefix caching.

All model-free: prefix matching is hash bookkeeping.

The parent-hash tests are the important ones. Without parent chaining you get a
cache that returns blocks whose K/V were computed under a different history --
silently wrong output, only on cache hits, only under specific traffic. It is
one of the nastiest bugs in this whole course, so it's pinned hard here.
"""

from __future__ import annotations

import pytest


def _bm(n_blocks: int = 64, block_size: int = 16):
    from engine.block_manager import BlockManager

    return BlockManager(n_blocks=n_blocks, block_size=block_size,
                        enable_prefix_caching=True)


def _seq(seq_id: int, token_ids: list[int]):
    from engine.sequence import Sequence

    return Sequence(seq_id=seq_id, prompt_ids=token_ids)


SYSTEM = list(range(1000, 1064))     # 64 tokens = 4 blocks of 16


class TestMatching:
    def test_cold_cache_matches_nothing(self):
        bm = _bm()
        blocks, n_hit = bm.match_prefix(SYSTEM)
        assert blocks == [] and n_hit == 0

    def test_identical_prompt_hits_fully(self):
        bm = _bm()
        a = _seq(0, SYSTEM)
        bm.allocate(a)
        bm.cache_blocks(a)                      # publish after prefill

        blocks, n_hit = bm.match_prefix(SYSTEM)
        assert n_hit == 64, f"expected a full 64-token hit, got {n_hit}"
        assert len(blocks) == 4

    def test_shared_prefix_hits_partially(self):
        """Common system prompt, different question."""
        bm = _bm()
        a = _seq(0, SYSTEM + [7, 7, 7])
        bm.allocate(a)
        bm.cache_blocks(a)

        _, n_hit = bm.match_prefix(SYSTEM + [9, 9, 9])
        assert n_hit == 64, "should reuse the shared 64-token prefix"

    def test_prefix_ends_at_first_difference(self):
        """The rule that decides your savings.

        Same tokens, novel content first -> nothing is reusable. This is
        bench/workloads.py's late_divergence, as an assertion.
        """
        bm = _bm()
        a = _seq(0, [42] + SYSTEM)
        bm.allocate(a)
        bm.cache_blocks(a)

        _, n_hit = bm.match_prefix([99] + SYSTEM)
        assert n_hit == 0, (
            "First token differs, so every later block's parent hash differs "
            "too. Identical trailing tokens are worthless."
        )

    def test_partial_blocks_are_not_cacheable(self):
        """A partially-filled block is still being written."""
        bm = _bm(block_size=16)
        a = _seq(0, list(range(20)))            # 1 full block + 4 tokens
        bm.allocate(a)
        bm.cache_blocks(a)

        _, n_hit = bm.match_prefix(list(range(20)))
        assert n_hit == 16, f"only the full block is cacheable, got {n_hit}"


class TestParentHashing:
    def test_same_tokens_different_history_do_not_match(self):
        """THE correctness test for prefix caching.

        Block content is not determined by its own tokens: K/V at position 16
        depend on tokens 0-15. Two blocks with identical tokens but different
        predecessors hold different K/V and must never be shared.
        """
        bm = _bm(block_size=16)
        shared_tail = list(range(500, 516))     # identical 16-token block

        a = _seq(0, list(range(0, 16)) + shared_tail)
        bm.allocate(a)
        bm.cache_blocks(a)

        # Same tail, different first block.
        blocks, n_hit = bm.match_prefix(list(range(100, 116)) + shared_tail)
        assert n_hit == 0, (
            "Matched a block with the same tokens but a different history. "
            "The parent hash must be part of each block's identity."
        )


class TestRefCounting:
    def test_shared_block_is_not_freed_early(self):
        bm = _bm()
        a = _seq(0, SYSTEM)
        bm.allocate(a)
        bm.cache_blocks(a)

        b = _seq(1, SYSTEM)
        shared, _ = bm.match_prefix(b.prompt_ids)
        b.block_table = list(shared)

        bm.free(a)                              # a leaves; b still needs these
        _, n_hit = bm.match_prefix(SYSTEM)
        assert n_hit == 64, "block freed while another sequence held it"

    def test_hit_does_not_double_allocate(self):
        """Cache hits must consume no new blocks."""
        bm = _bm(n_blocks=8, block_size=16)
        a = _seq(0, SYSTEM)
        bm.allocate(a)
        bm.cache_blocks(a)
        before = len(bm.free_blocks)

        bm.match_prefix(SYSTEM)
        assert len(bm.free_blocks) == before


class TestEviction:
    def test_refcount_zero_blocks_are_kept_until_needed(self):
        """A finished sequence's blocks stay cached -- that IS the cache."""
        bm = _bm(n_blocks=64)
        a = _seq(0, SYSTEM)
        bm.allocate(a)
        bm.cache_blocks(a)
        bm.free(a)

        _, n_hit = bm.match_prefix(SYSTEM)
        assert n_hit == 64, "cache should survive the sequence that filled it"

    def test_eviction_reclaims_under_pressure(self):
        """Small pool, many distinct prompts -> old entries must be reclaimed
        rather than the allocator failing."""
        bm = _bm(n_blocks=8, block_size=16)
        for i in range(20):
            seq = _seq(i, list(range(i * 1000, i * 1000 + 32)))
            assert bm.can_allocate(seq), f"could not allocate at {i}"
            bm.allocate(seq)
            bm.cache_blocks(seq)
            bm.free(seq)


class TestWorkloadsAgree:
    """The shipped workloads must actually exhibit the contrast."""

    def test_shared_prefix_and_late_divergence_differ(self):
        from bench.workloads import late_divergence, shared_prefix

        sp = shared_prefix(n=4).requests
        ld = late_divergence(n=4).requests

        sp_common = len(_common_prefix(sp[0].prompt, sp[1].prompt))
        ld_common = len(_common_prefix(ld[0].prompt, ld[1].prompt))

        assert sp_common > 500, "shared_prefix should share a long prefix"
        assert ld_common < 50, "late_divergence should share almost nothing"


def _common_prefix(a: str, b: str) -> str:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return a[:n]

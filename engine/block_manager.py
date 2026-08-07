"""M1.5 -> M1.6 -- block allocation and prefix caching.

M1.5 allocate/free fixed-size blocks; maintain each sequence's block table.
M1.6 hash block contents so identical prefixes share physical blocks. Needs
     reference counting (a block is freed only when nobody points at it) and
     an eviction policy (LRU over refcount-zero blocks).

The M1.6 lesson (book §5.3.1): a prefix ends at the FIRST differing token.
Run workloads.shared_prefix against workloads.late_divergence -- nearly the
same tokens, wildly different hit rates. Context ordering IS the optimization.

Compare with nano-vllm's engine/block_manager.py.
"""

from __future__ import annotations


class BlockManager:
    """M1.5 paged allocation; M1.6 adds content-hash prefix sharing."""

    def __init__(self, n_blocks: int, block_size: int = 16,
                 enable_prefix_caching: bool = False):
        raise NotImplementedError("M1.5")

    def allocate(self, sequence) -> list[int]:
        raise NotImplementedError("M1.5")

    def free(self, sequence) -> None:
        """Decrement refcounts; return blocks that reached zero to the pool."""
        raise NotImplementedError("M1.5")

    def match_prefix(self, token_ids: list[int]) -> tuple[list[int], int]:
        """M1.6 (Lecture 10). Return (cached_block_ids, n_tokens_hit).

        Walk the sequence block by block, hashing (parent_hash, tokens). Stop
        at the first miss -- once the chain breaks, no later block can match.

        The parent hash is not optional: K/V at position 16 depend on tokens
        0-15, so identical tokens with different histories are different
        blocks. Omitting it gives silently wrong output on cache hits.
        """
        raise NotImplementedError("M1.6")

    def cache_blocks(self, sequence) -> None:
        """M1.6. Publish a sequence's FULL blocks into the hash index.

        Called after prefill, when the contents are final. Partial blocks are
        still being written and must not be published.
        """
        raise NotImplementedError("M1.6")

    def evict(self) -> int | None:
        """M1.6. Reclaim the least-recently-used refcount-0 block.

        A refcount-0 block is not garbage -- it is cached, and may serve a
        future request. Reclaim only under memory pressure. Returns None when
        nothing is evictable, which means genuine exhaustion (preempt).
        """
        raise NotImplementedError("M1.6")

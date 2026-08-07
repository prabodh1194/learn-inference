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
        """M1.6. Return (cached_block_ids, n_tokens_hit)."""
        raise NotImplementedError("M1.6")

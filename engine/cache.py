"""M1.1 -> M1.5 -- the KV cache.

M1.1 KVCache        contiguous per-sequence tensors. Simple, and it wastes a
                    lot: you must reserve max_seq_len up front per sequence.
M1.5 PagedKVCache   fixed-size blocks + a block table mapping logical position
                    -> physical block. Allocate on demand.

The M1.5 number to move: max concurrent sequences before OOM. That memory win
is the single biggest result in Phase 1 (book §2.5, §5.3.2).
"""

from __future__ import annotations


class KVCache:
    """M1.1. Contiguous. Reserves max_seq_len per sequence -- note the waste."""

    def __init__(self, n_layers: int, max_seqs: int, max_seq_len: int,
                 n_kv_heads: int, head_dim: int, dtype=None, device=None):
        raise NotImplementedError("M1.1")


class PagedKVCache:
    """M1.5. Block-based. This is PagedAttention's core data structure."""

    def __init__(self, n_layers: int, n_blocks: int, block_size: int,
                 n_kv_heads: int, head_dim: int, dtype=None, device=None):
        raise NotImplementedError("M1.5")

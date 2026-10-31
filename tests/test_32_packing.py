"""Lecture 32 -- Sequence packing, cu_seqlens, and token-weighted loss."""

from __future__ import annotations

import pytest

from book.code.packing_waste import packed_cost, padded_cost, sample_doc_lengths
from train.packing import cu_seqlens_from_pack, pack_sequences, token_weighted_loss


class TestBookArithmetic:
    def test_packing_reduces_allocated_slots_over_2x(self):
        lengths = sample_doc_lengths()
        _, alloc_pad = padded_cost(lengths, batch_size=8)
        _, alloc_pack, _ = packed_cost(lengths, bin_len=2048)
        assert round(alloc_pad / alloc_pack, 2) == 2.28


class TestImplementation:
    def test_pack_sequences_preserves_all_docs(self):
        bins = pack_sequences([500, 600, 400, 700], max_seq_len=1024)
        assert sum(sum(b) for b in bins) == 2200
        assert all(sum(b) <= 1024 for b in bins)

    def test_cu_seqlens_starts_at_zero_and_accumulates(self):
        assert cu_seqlens_from_pack([120, 300, 80]) == [0, 120, 420, 500]

    def test_token_weighted_loss_fixes_microbatch_bias(self):
        got = token_weighted_loss(sum_losses=[200.0, 50.0], token_counts=[100, 10])
        assert got == pytest.approx(250.0 / 110.0)

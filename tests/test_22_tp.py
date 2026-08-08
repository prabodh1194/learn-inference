"""Lecture 22 -- tensor parallelism.

Sharding ARITHMETIC is testable on one device; scaling curves are not.

The column/row split test is the important one: get the axes backwards and
you need two all-reduces per MLP instead of one, doubling your communication.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestShardingMath:
    """Runs anywhere -- this is the part people get wrong on paper."""

    def test_column_parallel_splits_output_dim(self):
        from engine.parallel import shard_column

        w = torch.randn(512, 1024)          # (in, out)
        shards = [shard_column(w, rank=r, world_size=4) for r in range(4)]
        assert all(s.shape == (512, 256) for s in shards)
        torch.testing.assert_close(torch.cat(shards, dim=1), w)

    def test_row_parallel_splits_input_dim(self):
        from engine.parallel import shard_row

        w = torch.randn(1024, 512)          # (in, out)
        shards = [shard_row(w, rank=r, world_size=4) for r in range(4)]
        assert all(s.shape == (256, 512) for s in shards)
        torch.testing.assert_close(torch.cat(shards, dim=0), w)

    def test_column_then_row_reconstructs_the_mlp(self):
        """The elegance: column-parallel needs no comms, row-parallel's
        partial sums add up. One all-reduce per MLP, not two."""
        from engine.parallel import shard_column, shard_row

        torch.manual_seed(0)
        x = torch.randn(4, 512)
        w1 = torch.randn(512, 1024)
        w2 = torch.randn(1024, 512)
        want = (x @ w1) @ w2

        partials = [
            (x @ shard_column(w1, r, 4)) @ shard_row(w2, r, 4) for r in range(4)
        ]
        torch.testing.assert_close(sum(partials), want, rtol=1e-3, atol=1e-3)

    def test_attention_shards_by_head(self):
        from engine.parallel import shard_heads

        n_heads, head_dim = 16, 128
        w = torch.randn(1024, n_heads * head_dim)
        s = shard_heads(w, rank=0, world_size=4, head_dim=head_dim)
        assert s.shape == (1024, 4 * head_dim), "4 of 16 heads per rank"


class TestCommunicationCost:
    def test_allreduce_volume_is_independent_of_world_size(self):
        """Why scaling is sublinear: compute divides by N, comms doesn't."""
        from engine.parallel import allreduce_bytes

        vols = [allreduce_bytes(hidden=4096, tokens=32, world_size=n)
                for n in (2, 4, 8)]
        assert len(set(vols)) == 1, "all-reduce volume should not shrink with N"

    def test_predicted_speedup_is_sublinear(self):
        from engine.parallel import predicted_speedup

        s2 = predicted_speedup(compute_ms=10.0, comm_ms=1.0, world_size=2)
        s4 = predicted_speedup(compute_ms=10.0, comm_ms=1.0, world_size=4)
        assert s2 < 2.0 and s4 < 4.0
        assert s4 > s2, "more GPUs should still help, just not linearly"


@pytest.mark.cuda
@pytest.mark.slow
def test_tp_output_matches_single_gpu():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs 2+ GPUs")
    from engine.parallel import run_tp_reference_check

    assert run_tp_reference_check()

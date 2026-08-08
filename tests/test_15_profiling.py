"""Lecture 15 -- profiling.

The harness is testable without CUDA; the profiling itself is not.

A ranking that doesn't sum to ~100% or that omits the top kernels will send
you optimizing the wrong thing for a weekend, so it's worth asserting.
"""

from __future__ import annotations

import pytest


class TestRankingMath:
    """Pure arithmetic -- runs anywhere."""

    def test_shares_sum_to_one(self):
        from kernels.profile_engine import rank_kernels

        raw = {"matmul": 60.0, "attention": 25.0, "rmsnorm": 10.0, "sample": 5.0}
        ranked = rank_kernels(raw)
        assert abs(sum(share for _, _, share in ranked) - 1.0) < 1e-6

    def test_sorted_by_time_descending(self):
        from kernels.profile_engine import rank_kernels

        ranked = rank_kernels({"a": 5.0, "b": 60.0, "c": 25.0})
        assert [n for n, _, _ in ranked] == ["b", "c", "a"]

    def test_amdahl_bound(self):
        """The step people skip: what's the CEILING on this optimization?

        A kernel at 8% of runtime made 3x faster buys 5.3% end to end. If you
        can't state that number, you don't know whether it's worth a weekend.
        """
        from kernels.profile_engine import amdahl_speedup

        assert amdahl_speedup(share=0.08, speedup=3.0) == pytest.approx(1.0563, abs=1e-3)
        assert amdahl_speedup(share=0.60, speedup=2.0) == pytest.approx(1.4286, abs=1e-3)

    def test_infinite_speedup_is_bounded_by_share(self):
        """Even an infinitely fast kernel only removes its own share."""
        from kernels.profile_engine import amdahl_speedup

        assert amdahl_speedup(share=0.10, speedup=float("inf")) == pytest.approx(1 / 0.9)


class TestBandwidthUtilization:
    def test_fraction_of_peak(self):
        """Lecture 15's stop condition: near peak means look elsewhere."""
        from kernels.profile_engine import bandwidth_utilization

        # 936 GB/s peak, moved 9.36 GB in 0.0125s -> 748.8 GB/s -> 80%
        assert bandwidth_utilization(
            bytes_moved=9.36e9, seconds=0.0125, peak_bandwidth=936e9
        ) == pytest.approx(0.8, abs=0.01)


@pytest.mark.cuda
@pytest.mark.slow
def test_profile_produces_a_ranking(model_and_tokenizer):
    from kernels.profile_engine import profile_decode

    model, tok = model_and_tokenizer
    ranked = profile_decode(model, tok, batch_size=1, steps=8)
    assert ranked, "profiler returned no kernels"
    assert all(share >= 0 for _, _, share in ranked)

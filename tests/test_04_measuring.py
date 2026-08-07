"""Lecture 04 -- the measurement harness itself.

These pass today. They exist because every later number depends on this code
being right: if the ruler is wrong, every result in the course is wrong and
you'd have no way to tell.
"""

from __future__ import annotations

import time

import pytest

from bench.harness import Distribution, RequestRecord, percentile, summarize


class TestPercentile:
    def test_matches_hand_computed(self):
        assert percentile([1, 2, 3, 4, 5], 50) == 3
        assert percentile([1], 99) == 1

    def test_interpolates(self):
        # p50 of an even-length set falls between the two middle values
        assert percentile([1, 2, 3, 4], 50) == 2.5

    def test_tail_is_not_the_mean(self):
        """The reason the harness reports percentiles at all.

        A slow minority barely moves the mean but dominates the tail. A service
        tuned on means ships a bad p99, and nobody notices until users complain.
        """
        vals = [1.0] * 95 + [100.0] * 5
        d = Distribution.from_values(vals)
        assert d.p50 == 1.0          # the median says "fast"
        assert d.p99 > 50.0          # the tail says otherwise
        assert d.mean < 10.0         # and the mean hides both

    def test_p99_needs_enough_samples_to_mean_anything(self):
        """A caveat worth internalizing before you trust your own benchmarks.

        p99 interpolates between the top two samples, so with 100 requests a
        single outlier lands at ~p99 by construction and the estimate is noisy.
        Want a trustworthy p99? Send thousands of requests (Lecture 25).
        """
        one_outlier = Distribution.from_values([1.0] * 99 + [100.0])
        assert one_outlier.p99 < 10.0  # nowhere near the 100.0 outlier

    def test_empty_is_nan_not_crash(self):
        d = Distribution.from_values([])
        assert d.count == 0
        assert d.mean != d.mean  # NaN


class TestRequestRecord:
    def test_ttft_is_first_token_from_arrival(self):
        rec = RequestRecord(prompt_tokens=5)
        time.sleep(0.02)
        rec.mark_start()
        rec.mark_token()
        assert rec.ttft >= 0.02
        assert rec.queue_time >= 0.02

    def test_tpot_excludes_the_first_token(self):
        """TPOT measures the decode steady state.

        The first token includes prefill, which is a different phase with a
        different bottleneck. Averaging it in would blend compute-bound and
        memory-bound work into one meaningless number.
        """
        rec = RequestRecord()
        rec.mark_start()
        for _ in range(4):
            time.sleep(0.01)
            rec.mark_token()
        assert rec.output_tokens == 4
        assert len(rec.inter_token_latencies) == 3   # gaps, not tokens
        assert 0.005 < rec.tpot < 0.05

    def test_tpot_undefined_for_single_token(self):
        rec = RequestRecord()
        rec.mark_start()
        rec.mark_token()
        assert rec.tpot != rec.tpot  # NaN: no gap to measure


class TestSummarize:
    def test_aggregates_across_requests(self):
        recs = []
        for _ in range(3):
            r = RequestRecord(prompt_tokens=10)
            r.mark_start()
            for _ in range(5):
                time.sleep(0.002)
                r.mark_token()
            r.mark_end()
            recs.append(r)

        res = summarize(recs, name="t", milestone="M0.2", wall_time=1.0)
        assert res.n_requests == 3
        assert res.output_tokens == 15
        assert res.prompt_tokens == 30
        assert res.output_throughput == pytest.approx(15.0)
        assert res.itl.count == 12  # 3 requests x 4 gaps

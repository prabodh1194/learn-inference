"""Lecture 25 -- load testing.

The arithmetic of open-loop load generation. All model-free.

The Poisson test matters: a fixed inter-arrival time hides queueing entirely
and makes your service look far more predictable than it is.
"""

from __future__ import annotations

import statistics

import pytest


class TestArrivalPattern:
    def test_poisson_gaps_are_exponential(self):
        """Mean gap should be ~1/rate, and the gaps must VARY."""
        from bench.workloads import mixed_length, poisson_arrivals

        w = poisson_arrivals(mixed_length(n=400, seed=0), rate=20.0, seed=0)
        offsets = [r.arrival_offset for r in w]
        gaps = [b - a for a, b in zip(offsets, offsets[1:])]

        assert statistics.fmean(gaps) == pytest.approx(1 / 20.0, rel=0.25)
        assert statistics.pstdev(gaps) > 0.2 * statistics.fmean(gaps), (
            "gaps are nearly constant -- that's a fixed-rate loop, not Poisson, "
            "and it will hide queueing"
        )

    def test_offsets_are_monotonic(self):
        from bench.workloads import mixed_length, poisson_arrivals

        offs = [r.arrival_offset
                for r in poisson_arrivals(mixed_length(n=100), rate=10.0)]
        assert offs == sorted(offs)


class TestOpenLoop:
    def test_open_loop_does_not_self_limit(self):
        """The distinction that decides what you can observe.

        Closed loop: N clients each wait for a response, so offered load falls
        when the server slows -- overload is literally unobservable.
        Open loop: arrivals continue regardless, so the queue grows.
        """
        from bench.load_test import simulate

        slow = simulate(rate=100.0, service_time=0.05, duration=2.0, mode="open")
        assert slow["queue_depth_final"] > slow["queue_depth_initial"], (
            "an overloaded open-loop test must show a growing queue"
        )

    def test_closed_loop_queue_is_bounded_by_clients(self):
        from bench.load_test import simulate

        r = simulate(rate=100.0, service_time=0.05, duration=2.0,
                     mode="closed", clients=8)
        assert r["queue_depth_final"] <= 8


class TestKnee:
    def test_finds_where_throughput_saturates(self):
        from bench.load_test import find_knee

        # throughput plateaus at 100 while latency climbs
        points = [(10, 10, 0.10), (50, 50, 0.12), (90, 90, 0.20),
                  (120, 100, 0.90), (200, 100, 4.00)]
        knee = find_knee(points)
        assert 50 <= knee <= 120, f"knee at {knee}, expected around 90-120"

    def test_p99_degrades_faster_than_p50(self):
        """The tail always goes first -- which is why L04 insisted on
        percentiles rather than means."""
        from bench.load_test import simulate

        light = simulate(rate=10.0, service_time=0.05, duration=2.0, mode="open")
        heavy = simulate(rate=100.0, service_time=0.05, duration=2.0, mode="open")
        assert (heavy["p99"] / light["p99"]) > (heavy["p50"] / light["p50"])

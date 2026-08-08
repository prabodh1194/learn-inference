"""Lecture 27 -- routing and disaggregation.

Routing policy is pure logic. No GPU.
"""

from __future__ import annotations

import pytest


class TestRoutingPolicies:
    def test_round_robin_ignores_cache_state(self):
        """The baseline, and the problem: your prefix cache fragments."""
        from serve.router import Router

        r = Router(n_replicas=4, strategy="round_robin")
        picks = [r.route(prefix_hash="conversation-1") for _ in range(4)]
        assert len(set(picks)) == 4, "round robin should spread regardless"

    def test_session_affinity_is_stable(self):
        from serve.router import Router

        r = Router(n_replicas=4, strategy="session")
        picks = {r.route(prefix_hash="conversation-1") for _ in range(10)}
        assert len(picks) == 1, "same conversation must land on one replica"

    def test_cache_aware_prefers_a_warm_replica(self):
        from serve.router import Router

        r = Router(n_replicas=4, strategy="cache_aware")
        r.mark_warm(replica=2, prefix_hash="sys-prompt")
        assert r.route(prefix_hash="sys-prompt") == 2

    def test_cache_aware_avoids_an_overloaded_hotspot(self):
        """Route purely by affinity and you create hotspots. Every real
        router blends affinity with load."""
        from serve.router import Router

        r = Router(n_replicas=4, strategy="cache_aware", load_threshold=0.9)
        r.mark_warm(replica=2, prefix_hash="sys-prompt")
        r.set_load(replica=2, load=1.0)
        assert r.route(prefix_hash="sys-prompt") != 2

    def test_hit_rate_beats_round_robin_on_multiturn(self):
        from serve.router import Router, simulate_multiturn

        rr = simulate_multiturn(Router(n_replicas=4, strategy="round_robin"))
        ca = simulate_multiturn(Router(n_replicas=4, strategy="cache_aware"))
        assert ca["hit_rate"] > rr["hit_rate"]


class TestDisaggregation:
    def test_transfer_cost_scales_with_prompt_length(self):
        from serve.disaggregated import kv_transfer_bytes

        short = kv_transfer_bytes(prompt_tokens=64)
        long = kv_transfer_bytes(prompt_tokens=4096)
        assert long == pytest.approx(short * 64, rel=0.01)

    def test_disaggregation_loses_on_short_prompts(self):
        """It is a judgment call, not a strict improvement."""
        from serve.disaggregated import is_worthwhile

        assert not is_worthwhile(prompt_tokens=32, interconnect_gbps=100,
                                 contention=0.5)
        assert is_worthwhile(prompt_tokens=8192, interconnect_gbps=100,
                             contention=0.9)

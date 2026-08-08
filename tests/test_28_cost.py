"""Lecture 28 -- autoscaling and cost.

The reframe: at low utilization, utilization dominates every kernel
optimization in Part III. Pure arithmetic, and it should change your
priorities.
"""

from __future__ import annotations

import pytest


class TestCostModel:
    def test_cost_per_million_tokens(self):
        from bench.cost_model import cost_per_million

        # $0.25/hr at 2000 tok/s = 7.2M tokens/hr -> ~$0.0347/M
        assert cost_per_million(dollars_per_hour=0.25, tokens_per_second=2000,
                                utilization=1.0) == pytest.approx(0.0347, abs=0.001)

    def test_low_utilization_dominates(self):
        """The punchline of Part V: same engine, 5x the cost."""
        from bench.cost_model import cost_per_million

        full = cost_per_million(0.25, 2000, utilization=1.0)
        fifth = cost_per_million(0.25, 2000, utilization=0.2)
        assert fifth == pytest.approx(full * 5, rel=0.01)

    def test_utilization_beats_a_kernel_win(self):
        """A 30% faster engine at 20% utilization still loses to the same
        engine at 50%. Know which lever you're pulling."""
        from bench.cost_model import cost_per_million

        fast_but_idle = cost_per_million(0.25, 2600, utilization=0.2)
        slow_but_busy = cost_per_million(0.25, 2000, utilization=0.5)
        assert slow_but_busy < fast_but_idle


class TestAutoscaling:
    def test_scales_on_queue_depth(self):
        from serve.autoscale import decide

        assert decide(queue_depth=50, replicas=2, target_queue=5) > 2
        assert decide(queue_depth=0, replicas=4, target_queue=5) < 4

    def test_respects_cooldown(self):
        """Cold starts take minutes; flapping is expensive."""
        from serve.autoscale import Autoscaler

        a = Autoscaler(target_queue=5, cooldown_s=60)
        a.observe(queue_depth=50, now=0.0)
        first = a.desired_replicas
        a.observe(queue_depth=0, now=10.0)      # inside cooldown
        assert a.desired_replicas == first

    def test_gpu_utilization_is_a_poor_signal(self):
        """A memory-bound decode loop shows high GPU utilization while doing
        very little work -- which is why you scale on queue depth."""
        from serve.autoscale import is_reliable_signal

        assert not is_reliable_signal("gpu_utilization")
        assert is_reliable_signal("queue_depth")

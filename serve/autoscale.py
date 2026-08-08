"""M4.6 (Lecture 28) -- autoscaling.

Scale on the right signal. CPU utilization is meaningless here, and GPU
utilization is actively misleading: a memory-bound decode loop shows high
utilization while doing very little work. Scale on QUEUE DEPTH, the honest
saturation signal from Lecture 25.

And respect cold starts. Provisioning a node, pulling a multi-gigabyte
container, loading weights, and warming CUDA graphs takes MINUTES. Flapping is
expensive, which is what the cooldown is for.
"""

from __future__ import annotations

# Signals that actually track saturation for LLM serving.
_RELIABLE = {"queue_depth", "concurrent_sequences", "time_in_queue"}
_MISLEADING = {"gpu_utilization", "cpu_utilization", "memory_used"}


def is_reliable_signal(signal: str) -> bool:
    """Implemented -- the taxonomy is the lesson, not the code.

    gpu_utilization is the trap: it reads high during memory-bound decode
    regardless of how much useful work is happening.
    """
    if signal in _RELIABLE:
        return True
    if signal in _MISLEADING:
        return False
    raise ValueError(f"unknown signal: {signal}")


def decide(queue_depth: int, replicas: int, target_queue: int) -> int:
    """M4.6. Desired replica count for the observed queue depth."""
    raise NotImplementedError("M4.6")


class Autoscaler:
    """M4.6. Queue-depth scaling with a cooldown to prevent flapping."""

    def __init__(self, target_queue: int = 5, cooldown_s: float = 60.0):
        raise NotImplementedError("M4.6")

    def observe(self, queue_depth: int, now: float) -> None:
        raise NotImplementedError("M4.6")

    @property
    def desired_replicas(self) -> int:
        raise NotImplementedError("M4.6")

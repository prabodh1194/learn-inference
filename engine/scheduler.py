"""M1.4 -> M1.7 -- the scheduler. The first genuinely new idea in the plan.

Static batching fixes the batch for its whole lifetime, so a short request
waits for the longest one to finish. Continuous batching admits and retires
sequences *per step*, which requires splitting generation into:

    scheduler   decides which requests run this step
    model_runner executes exactly one step

That split is the architecture of every real engine. Compare yours to
nano-vllm's engine/scheduler.py after M1.4.

M1.4 measure on workloads.mixed_length -- that is where it wins.
M1.7 chunked prefill: split long prefills and interleave with decode. Judge on
     p99 of the SHORT requests in workloads.long_prefill, not the mean.
"""

from __future__ import annotations

from enum import Enum


class Status(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


class Scheduler:
    """M1.4. Continuous batching; M1.7 adds chunked prefill."""

    def __init__(self, max_batch_size: int = 32, max_batched_tokens: int = 8192,
                 chunked_prefill: bool = False, chunk_size: int = 512):
        raise NotImplementedError("M1.4")

    def add(self, request) -> None:
        raise NotImplementedError("M1.4")

    def schedule(self) -> list:
        """Pick this step's batch. Must handle admission, preemption, completion."""
        raise NotImplementedError("M1.4")

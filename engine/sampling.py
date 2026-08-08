"""M1.2 -- sampling.

Cheap to write, but subtle bugs hide here and they will confuse every later
benchmark. Verify greedy (temperature=0) is bit-for-bit deterministic across
runs before trusting anything downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 0          # 0 = disabled
    top_p: float = 1.0      # 1.0 = disabled
    repetition_penalty: float = 1.0
    max_tokens: int = 128
    seed: int | None = None

    # Lecture 12b: called with the logits before sampling; returns masked
    # logits. This one hook is how guided decoding, tool calling, and any
    # custom constraint plug in.
    logit_processor: "Callable[[object], object] | None" = None

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


def sample(logits, params: SamplingParams, prev_tokens=None):
    """Apply penalties, then temperature, then top-k/top-p; return token ids."""
    raise NotImplementedError("M1.2")

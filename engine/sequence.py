"""M1.4 (Lecture 08) -- per-request state.

Once sequences enter and leave the batch independently, "the batch" stops being
a meaningful unit of state. Each request carries its own.

Lecture 09 extends this with `block_table` -- where this sequence's KV cache
actually lives in physical memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Status(Enum):
    WAITING = "waiting"      # queued, not yet admitted
    RUNNING = "running"      # in the batch, generating
    PREEMPTED = "preempted"  # evicted under memory pressure (L09)
    FINISHED = "finished"    # hit EOS or max_tokens


@dataclass
class Sequence:
    """One request, from arrival to completion."""

    seq_id: int
    prompt_ids: list[int]
    output_ids: list[int] = field(default_factory=list)
    status: Status = Status.WAITING
    max_tokens: int = 128
    eos_token_id: int | None = None

    # Lecture 09: physical blocks holding this sequence's KV cache.
    block_table: list[int] = field(default_factory=list)

    # Lecture 11: how much of the prompt has been prefilled so far. With
    # chunked prefill a long prompt is processed across several steps.
    num_prefilled: int = 0

    def __len__(self) -> int:
        """Total context length -- what the KV cache must hold."""
        return len(self.prompt_ids) + len(self.output_ids)

    @property
    def is_prefill_done(self) -> bool:
        return self.num_prefilled >= len(self.prompt_ids)

    def is_finished(self) -> bool:
        if len(self.output_ids) >= self.max_tokens:
            return True
        if self.eos_token_id is not None and self.output_ids:
            return self.output_ids[-1] == self.eos_token_id
        return False

    def append(self, token_id: int) -> None:
        self.output_ids.append(token_id)

    def all_ids(self) -> list[int]:
        return self.prompt_ids + self.output_ids

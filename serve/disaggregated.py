"""M4.5 (Lecture 27) -- prefill/decode disaggregation.

Prefill is compute-bound, decode is memory-bound (Lecture 01). On one GPU they
compete and tuning for one hurts the other -- so run them on separate workers.

The cost is transferring the KV cache between them, which for a long prompt is
gigabytes. Whether it wins depends on whether that transfer costs less than
the interference it removes. It is a judgment call, not a strict improvement.
"""

from __future__ import annotations

from book.code.roofline import ModelDims


def kv_transfer_bytes(prompt_tokens: int, dims: ModelDims | None = None) -> int:
    """Bytes moved from the prefill worker to the decode worker.

    Linear in prompt length -- which is exactly why short prompts lose.
    """
    d = dims or ModelDims()
    return d.kv_bytes_per_token() * prompt_tokens


def is_worthwhile(prompt_tokens: int, interconnect_gbps: float,
                  contention: float) -> bool:
    """M4.5. Does disaggregation win for this request?

    Wins when: prompts are long, contention is high, interconnect is fast.
    Loses when: prompts are short (transfer dominates a cheap prefill), load
    is light (no interference to remove), or the interconnect is slow.

    Conditional disaggregation -- short prompts straight to decode, long ones
    split -- is common for this reason.
    """
    raise NotImplementedError("M4.5")


class PrefillWorker:
    """M4.5. Computes the KV cache and the first token, then hands off."""

    def __init__(self, model):
        raise NotImplementedError("M4.5")


class DecodeWorker:
    """M4.5. Receives a KV cache and generates the remaining tokens."""

    def __init__(self, model):
        raise NotImplementedError("M4.5")

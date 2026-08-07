"""M0.1 -> M1.3 -- the generation loop, in three stages.

M0.1 generate_naive     no cache; recompute the full forward every token.
                        Deliberately slow. Feel the pain before fixing it.
M1.1 generate_cached    add the KV cache. Re-run bench/plot.time_per_token_curve
                        against M0.1: the climbing curve should go flat.
M1.3 generate_batched   pad a batch to uniform length and run it together.
                        Measure BOTH the throughput gain and the padding waste --
                        the waste is what motivates M1.4 continuous batching.
"""

from __future__ import annotations

from typing import Callable

# Called once per generated token so the benchmark harness can timestamp it.
# Pass RequestRecord.mark_token here.
OnToken = Callable[[], None] | None


def generate_naive(model, tokenizer, prompt: str, max_tokens: int = 128,
                   on_token: OnToken = None) -> str:
    """M0.1. No KV cache. O(n^2) total work -- that is the point.

    Sketch:
        tokens = tokenizer(prompt, return_tensors="pt").input_ids
        for _ in range(max_tokens):
            logits = model(tokens).logits      # the WHOLE sequence, every time
            next_id = logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, next_id], dim=-1)
            if on_token: on_token()
    """
    raise NotImplementedError("M0.1")


def generate_cached(model, tokenizer, prompt: str, max_tokens: int = 128,
                    on_token: OnToken = None) -> str:
    """M1.1. KV cache: prefill once, then feed one token at a time.

    Re-run notes/00-baseline/m03_curve.py and overlay on M0.1 -- the climbing
    curve should go flat. That contrast is the milestone.
    """
    raise NotImplementedError("M1.1")


def generate_batched(model, tokenizer, prompts: list[str], max_tokens: int = 128,
                     on_token: OnToken = None) -> list[str]:
    """M1.3. Static batching: pad to uniform length, generate together.

    Report padding_waste_fraction alongside throughput -- the waste is what
    motivates continuous batching (M1.4).
    """
    raise NotImplementedError("M1.3")

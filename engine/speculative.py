"""M1.8 -- speculative decoding.

Start with n-gram / prompt-lookup speculation: no draft model, so no training,
and the whole draft->verify->accept loop is visible in a few dozen lines.

Report acceptance_rate alongside tok/s. A speedup with a low acceptance rate
means you got lucky on batch size; a high acceptance rate that does NOT speed
things up means verification overhead is eating the win. Both are worth seeing.

The lesson is the contrast: run workloads.code_completion (output echoes input,
n-grams hit) against workloads.prose (novel output, acceptance collapses).
Book §5.2.4.
"""

from __future__ import annotations


class NgramSpeculator:
    """M1.8. Build an n-gram map from the prompt, propose suffixes as drafts."""

    def __init__(self, n: int = 3, max_draft_tokens: int = 8):
        raise NotImplementedError("M1.8")

    def propose(self, token_ids: list[int]) -> list[int]:
        raise NotImplementedError("M1.8")


def verify(target_logits, draft_tokens: list[int]) -> tuple[list[int], int]:
    """Accept the longest correct draft prefix. Return (accepted, n_accepted)."""
    raise NotImplementedError("M1.8")

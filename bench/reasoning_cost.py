"""L28b -- reasoning and test-time compute.

Reasoning models spend thousands of "thinking" tokens, and each is ordinary
decode (L01): the whole weight set is re-read per token. This computes the cost
per answer when accuracy is bought with tokens (longer traces, majority voting)
rather than with parameters.

The number that must move: $ per answer as a function of trace length and vote
count, and the params x tokens substitution -- a 32B model thinking ~21x longer
costs the same decode as a 671B model answering once.
"""

from __future__ import annotations


def reasoning_cost(params: int, tokens: int, bytes_per_param: int,
                   bandwidth: float, usd_per_hour: float) -> float:
    """Decode cost of one trace of `tokens` tokens from a model with `params`
    active parameters.

    bytes moved = tokens * params * bytes_per_param; time = bytes / bandwidth;
    cost = time/3600 * usd_per_hour. Returns dollars.
    """
    raise NotImplementedError("L28b")


def params_tokens_tradeoff(params_small: int, params_large: int) -> float:
    """How many more tokens the small model may spend at equal decode cost.

    Decode cost scales with params x tokens, so return
    params_large / params_small -- the token budget multiplier the small model
    "earns" by being cheaper per token.
    """
    raise NotImplementedError("L28b")

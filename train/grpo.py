"""L39 -- Group Relative Policy Optimization (GRPO) and verifiable rewards."""

from __future__ import annotations


def group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """L39. Normalize group rewards: (r_i - mean) / (std + eps) using population std."""
    raise NotImplementedError("L39")


def k3_kl_estimator(pi_prob: float, ref_prob: float) -> float:
    """L39. Schulman k3 unbiased non-negative KL estimator: (ref/pi) - log(ref/pi) - 1."""
    raise NotImplementedError("L39")


def grpo_token_loss(
    ratio: float,
    advantage: float,
    pi_prob: float,
    ref_prob: float,
    eps: float = 0.2,
    beta: float = 0.001,
) -> float:
    """L39. -(clipped_surrogate - beta * k3_kl) for a single token."""
    raise NotImplementedError("L39")

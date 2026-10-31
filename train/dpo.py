"""L38 -- Direct Preference Optimization (DPO)."""

from __future__ import annotations


def implicit_reward(logp_pi: float, logp_ref: float, beta: float = 0.1) -> float:
    """L38. beta * (logp_pi - logp_ref)."""
    raise NotImplementedError("L38")


def dpo_loss(
    logp_pi_w: float,
    logp_pi_l: float,
    logp_ref_w: float,
    logp_ref_l: float,
    beta: float = 0.1,
) -> float:
    """L38. -log(sigmoid(r_hat(w) - r_hat(l)))."""
    raise NotImplementedError("L38")


def dpo_gradient_weight(
    logp_pi_w: float,
    logp_pi_l: float,
    logp_ref_w: float,
    logp_ref_l: float,
    beta: float = 0.1,
) -> float:
    """L38. Per-pair wrongness weight sigmoid(r_hat(l) - r_hat(w))."""
    raise NotImplementedError("L38")

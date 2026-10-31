"""L37 -- Bradley-Terry reward modeling and PPO RLHF."""

from __future__ import annotations


def bradley_terry_loss(r_chosen: float, r_rejected: float) -> float:
    """L37. -log(sigmoid(r_chosen - r_rejected))."""
    raise NotImplementedError("L37")


def ppo_actor_critic_bytes(params: int) -> int:
    """L37. Total model-state bytes for Actor(16P) + Critic(16P) + Ref(2P) + RM(2P) = 36P."""
    raise NotImplementedError("L37")


def ppo_clipped_surrogate(
    ratio: float, advantage: float, eps: float = 0.2
) -> float:
    """L37. min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)."""
    raise NotImplementedError("L37")

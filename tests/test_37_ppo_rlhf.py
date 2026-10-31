"""Lecture 37 -- Bradley-Terry reward modeling and PPO RLHF."""

from __future__ import annotations

import math

import pytest

from train.ppo_rlhf import (
    bradley_terry_loss,
    ppo_actor_critic_bytes,
    ppo_clipped_surrogate,
)


class TestBookArithmetic:
    def test_ppo_four_models_cost_36_psi(self):
        params = 596_049_920
        gib = (36 * params) / (1024**3)
        assert round(gib, 2) == 19.98


class TestImplementation:
    def test_bradley_terry_shift_invariance(self):
        assert bradley_terry_loss(2.0, 2.0) == pytest.approx(math.log(2.0))
        assert bradley_terry_loss(5.0, 1.0) == pytest.approx(bradley_terry_loss(105.0, 101.0))

    def test_ppo_actor_critic_bytes(self):
        assert ppo_actor_critic_bytes(1000) == 36_000

    def test_ppo_clipped_surrogate_both_signs(self):
        assert ppo_clipped_surrogate(ratio=2.0, advantage=1.0, eps=0.2) == pytest.approx(1.2)
        assert ppo_clipped_surrogate(ratio=0.1, advantage=-1.0, eps=0.2) == pytest.approx(-0.8)

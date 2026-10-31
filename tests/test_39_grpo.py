"""Lecture 39 -- GRPO group-relative advantages and Schulman k3 KL."""

from __future__ import annotations

import pytest

from train.grpo import group_advantages, grpo_token_loss, k3_kl_estimator


class TestBookArithmetic:
    def test_worked_four_sample_advantages(self):
        rewards = [1.0, 0.0, 0.8, 0.2]
        mean = sum(rewards) / 4
        std = (sum((r - mean) ** 2 for r in rewards) / 4) ** 0.5
        a1 = (1.0 - mean) / std
        assert round(a1, 2) == 1.21


class TestImplementation:
    def test_group_advantages_zero_mean_and_unit_std(self):
        adv = group_advantages([1.0, 0.0, 0.8, 0.2])
        assert sum(adv) == pytest.approx(0.0, abs=1e-7)
        assert adv[0] == pytest.approx(-adv[1])

    def test_k3_kl_estimator_always_non_negative(self):
        assert k3_kl_estimator(0.2, 0.2) == pytest.approx(0.0)
        assert k3_kl_estimator(0.4, 0.2) > 0.0
        assert k3_kl_estimator(0.05, 0.2) > 0.0

    def test_grpo_token_loss_sign(self):
        pos = grpo_token_loss(ratio=1.1, advantage=1.0, pi_prob=0.2, ref_prob=0.2)
        neg = grpo_token_loss(ratio=1.1, advantage=-1.0, pi_prob=0.2, ref_prob=0.2)
        assert pos < 0.0 < neg

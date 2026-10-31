"""Lecture 38 -- Direct Preference Optimization (DPO)."""

from __future__ import annotations

import math

import pytest

from train.dpo import dpo_gradient_weight, dpo_loss


class TestBookArithmetic:
    def test_partition_function_cancels_in_margin(self):
        z_x = 999.0
        rw = (0.1 * (-10.0 - (-12.0))) + z_x
        rl = (0.1 * (-15.0 - (-14.0))) + z_x
        assert (rw - rl) == pytest.approx(0.3)


class TestImplementation:
    def test_dpo_at_init_is_ln2_with_half_weight(self):
        loss = dpo_loss(-12.0, -50.0, -12.0, -50.0, beta=0.1)
        wt = dpo_gradient_weight(-12.0, -50.0, -12.0, -50.0, beta=0.1)
        assert loss == pytest.approx(math.log(2.0))
        assert wt == pytest.approx(0.5)

    def test_dpo_weights_wrong_pairs_higher_than_right_pairs(self):
        w_wrong = dpo_gradient_weight(-14.0, -11.0, -12.0, -15.0, beta=0.1)
        w_right = dpo_gradient_weight(-8.0, -20.0, -12.0, -15.0, beta=0.1)
        assert w_wrong > 0.5 > w_right

"""Lecture 34 -- TP + Sequence Parallelism and Pipeline schedules."""

from __future__ import annotations

import pytest

from train.pipeline import (
    inflight_microbatches,
    pipeline_bubble_fraction,
    sp_activation_bytes,
)


class TestBookArithmetic:
    def test_sp_divides_full_34sbh_by_tp(self):
        t = 8
        tp_only = 10.0 + 24.0 / t
        tp_sp = 34.0 / t
        assert round(tp_only / tp_sp, 2) == 3.06


class TestImplementation:
    def test_sp_activation_bytes(self):
        no_sp = sp_activation_bytes(1000, 2400, tp_degree=8, use_sp=False)
        with_sp = sp_activation_bytes(1000, 2400, tp_degree=8, use_sp=True)
        assert no_sp == 1300
        assert with_sp == 425

    def test_pipeline_bubble_fraction(self):
        assert pipeline_bubble_fraction(pp_degree=4, n_microbatches=4) == pytest.approx(3 / 7)
        assert pipeline_bubble_fraction(pp_degree=4, n_microbatches=64) == pytest.approx(3 / 67)

    def test_1f1b_bounds_inflight_activations_at_p(self):
        assert inflight_microbatches(pp_degree=4, n_microbatches=64, schedule="gpipe") == 64
        assert inflight_microbatches(pp_degree=4, n_microbatches=64, schedule="1f1b") == 4

"""Lecture 30 -- The training ledger: 6ND FLOPs, 16*Psi bytes, and MFU."""

from __future__ import annotations

import pytest

from book.code.training_ledger import NON_EMB_PARAMS, TOTAL_PARAMS, bf16_round
from train.ledger import adamw_state_bytes, mfu, training_flops


class TestBookArithmetic:
    def test_qwen_16psi_is_8_point_88_gib(self):
        state_bytes = 16 * TOTAL_PARAMS
        assert round(state_bytes / (1024**3), 2) == 8.88
        assert state_bytes == 8 * (2 * TOTAL_PARAMS)

    def test_6nd_is_three_times_prefill(self):
        tokens = 4096
        fwd = 2 * NON_EMB_PARAMS * tokens
        bwd = 4 * NON_EMB_PARAMS * tokens
        assert fwd + bwd == 3 * fwd

    def test_bf16_stagnates_without_master_weights(self):
        w = bf16_round(1.0)
        for _ in range(1000):
            w = bf16_round(w - 1e-4)
        assert w == 1.0


class TestImplementation:
    def test_training_flops_6nd_and_8nd(self):
        assert training_flops(params=100, tokens=10, checkpointed=False) == 6000
        assert training_flops(params=100, tokens=10, checkpointed=True) == 8000

    def test_adamw_state_bytes_16psi(self):
        assert adamw_state_bytes(params=1000, master_fp32=True, grad_bytes=2) == 16_000
        assert adamw_state_bytes(params=1000, master_fp32=True, grad_bytes=4) == 18_000

    def test_mfu_ratio(self):
        assert mfu(35.5e12, 71.0e12) == pytest.approx(0.5)

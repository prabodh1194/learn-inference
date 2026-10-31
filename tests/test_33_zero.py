"""Lecture 33 -- DDP, ZeRO-1/2/3, and FSDP memory and communication."""

from __future__ import annotations

import pytest

from book.code.zero_memory import comm_gib
from train.zero import comm_bytes_per_step, zero_state_bytes


class TestBookArithmetic:
    def test_zero1_and_zero2_match_ddp_comm_volume(self):
        c0 = comm_gib(7e9, dp=8, stage=0)
        c1 = comm_gib(7e9, dp=8, stage=1)
        c2 = comm_gib(7e9, dp=8, stage=2)
        c3 = comm_gib(7e9, dp=8, stage=3)
        assert c0 == c1 == c2
        assert c3 == pytest.approx(1.5 * c0)


class TestImplementation:
    def test_zero_state_bytes_monotonic_decrease(self):
        s0 = zero_state_bytes(params=8000, dp_degree=8, stage=0)
        s1 = zero_state_bytes(params=8000, dp_degree=8, stage=1)
        s2 = zero_state_bytes(params=8000, dp_degree=8, stage=2)
        s3 = zero_state_bytes(params=8000, dp_degree=8, stage=3)
        assert s0 > s1 > s2 > s3
        assert s0 == 8 * s3

    def test_comm_bytes_per_step(self):
        assert comm_bytes_per_step(1000, stage=1) == 4000
        assert comm_bytes_per_step(1000, stage=2) == 4000
        assert comm_bytes_per_step(1000, stage=3) == 6000

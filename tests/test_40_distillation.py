"""Lecture 40 -- Reasoning distillation and rejection sampling."""

from __future__ import annotations

import pytest
import torch

from book.code.distillation_loss import softmax
from train.distill import kd_kl_loss, rejection_sample_filter


class TestBookArithmetic:
    def test_t2_scaling_stabilizes_kl_across_temperatures(self):
        zt, zs = [4.0, 2.0, 0.0, -1.0], [3.0, 2.5, 0.2, -0.5]
        pt1, ps1 = softmax(zt, 1.0), softmax(zs, 1.0)
        pt10, ps10 = softmax(zt, 10.0), softmax(zs, 10.0)
        kl1 = sum(p * torch.log(torch.tensor(p / q)).item() for p, q in zip(pt1, ps1, strict=True))
        kl10 = sum(p * torch.log(torch.tensor(p / q)).item() for p, q in zip(pt10, ps10, strict=True))
        assert kl10 < kl1 / 50
        assert 0.15 < (10.0**2) * kl10 < 0.30


class TestImplementation:
    def test_kd_kl_loss_zero_on_identical_logits(self):
        z = torch.tensor([[2.0, 1.0, -1.0]])
        assert kd_kl_loss(z, z, temperature=4.0) == pytest.approx(0.0, abs=1e-6)

    def test_rejection_sample_filter(self):
        items = [("bad", 0.0), ("good1", 1.0), ("good2", 1.0)]
        assert rejection_sample_filter(items, threshold=1.0) == ["good1", "good2"]

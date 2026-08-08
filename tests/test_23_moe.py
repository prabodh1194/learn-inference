"""Lecture 23 -- MoE and expert parallelism.

Routing is testable without a GPU, and the total-vs-active distinction is
where people go wrong: a 397B-A3B model needs VRAM for 397B.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class TestRouting:
    def test_topk_selects_k_experts(self):
        from jaxlm.moe import route

        logits = torch.randn(32, 128)          # 32 tokens, 128 experts
        idx, weights = route(logits, top_k=2)
        assert idx.shape == (32, 2)
        torch.testing.assert_close(weights.sum(-1), torch.ones(32),
                                   rtol=1e-4, atol=1e-4)

    def test_selects_the_highest_scoring(self):
        from jaxlm.moe import route

        logits = torch.tensor([[0.0, 5.0, 0.0, 3.0]])
        idx, _ = route(logits, top_k=2)
        assert set(idx[0].tolist()) == {1, 3}


class TestParameterAccounting:
    """The distinction that trips everyone."""

    def test_active_params_are_a_fraction_of_total(self):
        from jaxlm.moe import active_params, total_params

        cfg = dict(n_experts=128, top_k=2, expert_params=10_000_000,
                   dense_params=1_000_000)
        assert total_params(**cfg) == 128 * 10_000_000 + 1_000_000
        assert active_params(**cfg) == 2 * 10_000_000 + 1_000_000

    def test_memory_is_sized_by_total_not_active(self):
        """You must hold every expert -- the router might pick any of them."""
        from jaxlm.moe import active_params, total_params, vram_bytes

        cfg = dict(n_experts=128, top_k=2, expert_params=10_000_000,
                   dense_params=1_000_000)
        assert vram_bytes(**cfg, bytes_per_param=2) == total_params(**cfg) * 2
        assert vram_bytes(**cfg, bytes_per_param=2) > active_params(**cfg) * 2

    def test_moe_has_higher_intensity_than_dense(self):
        """The argument for MoE: fewer bytes moved per token at equal total
        size, so decode gets relatively cheaper."""
        from jaxlm.moe import active_params, total_params

        cfg = dict(n_experts=128, top_k=2, expert_params=10_000_000,
                   dense_params=1_000_000)
        moe_bytes = active_params(**cfg) * 2       # only active experts read
        dense_bytes = total_params(**cfg) * 2      # a dense model reads all
        assert moe_bytes < dense_bytes / 10


class TestLoadBalance:
    def test_imbalance_is_measurable(self):
        from jaxlm.moe import expert_load

        idx = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 1]])   # expert 0 is hot
        load = expert_load(idx, n_experts=4)
        assert load[0] == 4
        assert load.max() > load.float().mean(), "should detect the hotspot"

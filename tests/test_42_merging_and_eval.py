from __future__ import annotations
import math

from book.code.context_parallel_balance import (
    ep_all_to_all_mib,
    naive_ring_causal_blocks,
    zigzag_ring_causal_blocks,
)
from book.code.merging_and_eval import overopt_gold_reward, pass_at_k, ties_merge_1d
from book.code.rejection_sampling import (
    bon_kl_divergence,
    select_top_k_overall,
    select_top_per_prompt,
)
from book.code.scaling_laws import chinchilla_loss, lifetime_flops, yarn_attn_scale


class TestBookArithmetic:
    def test_l30b_overtraining_cuts_lifetime_flops(self) -> None:
        chinchilla = lifetime_flops(2.8e9, 56.0e9, 100.0e12)
        overtrained = lifetime_flops(1.4e9, 420.0e9, 100.0e12)
        assert overtrained["total_flops"] < chinchilla["total_flops"] * 0.55
        assert chinchilla_loss(1.4e9, 420.0e9) < chinchilla_loss(2.8e9, 56.0e9)
        assert math.isclose(yarn_attn_scale(8.0), 0.1 * math.log(8.0) + 1.0, rel_tol=1e-6)

    def test_l34b_zigzag_ring_balances_causal_flops(self) -> None:
        naive = naive_ring_causal_blocks(8)
        zigzag = zigzag_ring_causal_blocks(8)
        assert max(naive) / min(naive) == 15.0
        assert max(zigzag) == min(zigzag) == 8.0
        assert math.isclose(ep_all_to_all_mib(4096, 2048, 8, 8), 224.0, rel_tol=1e-6)

    def test_l36b_bon_kl_and_rejection_sampling(self) -> None:
        assert math.isclose(bon_kl_divergence(1), 0.0, abs_tol=1e-9)
        assert math.isclose(bon_kl_divergence(16), math.log(16) - 15.0 / 16.0, rel_tol=1e-6)
        rewards = [[0.9, 0.8], [0.4, 0.5]]
        per_p = select_top_per_prompt(rewards)
        top_k = select_top_k_overall(rewards, k=2)
        assert len({item[0] for item in per_p}) == 2
        assert len({item[0] for item in top_k}) == 1

    def test_l42_overoptimization_merging_and_pass_metrics(self) -> None:
        assert overopt_gold_reward(2.0, alpha=1.0, beta=0.25) == 1.0
        assert overopt_gold_reward(4.0, alpha=1.0, beta=0.25) == 0.0
        merged = ties_merge_1d([[0.6, -0.4], [0.4, 0.1], [-0.2, -0.6]])
        assert math.isclose(merged[0], 0.5, rel_tol=1e-6)
        assert math.isclose(merged[1], -0.5, rel_tol=1e-6)
        assert math.isclose(pass_at_k(10, 8, 5), 1.0, rel_tol=1e-6)
        assert math.isclose((8 / 10) ** 5, 0.32768, rel_tol=1e-6)

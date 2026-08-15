"""Lecture 28b -- reasoning and test-time compute.

Pure arithmetic: cost per answer scales with params x tokens, and majority
voting multiplies tokens (hence cost) linearly while accuracy climbs only
log-ish. No model needed.
"""

from __future__ import annotations


def test_ten_thousand_tokens_of_thinking_is_ten_thousand_weight_reads():
    """One token = one weight re-read. A 10k-token trace moves 10k x 840 MiB."""
    weight_read_mib = 840
    trace_tokens = 10_000
    total_mib = weight_read_mib * trace_tokens
    assert total_mib == 8_400_000
    assert total_mib / (1024 ** 2) == 8.0108642578125, "8 TiB of weight traffic"


def test_majority_voting_costs_linearly():
    """64 samples = 64x the tokens = 64x the decode cost."""
    assert 64 * 1 == 64


def test_params_tokens_tradeoff():
    from bench.reasoning_cost import params_tokens_tradeoff

    ratio = params_tokens_tradeoff(params_small=32_000_000_000,
                                   params_large=671_000_000_000)
    assert ratio == 671 / 32
    assert 20 < ratio < 21, "32B can think ~21x longer at equal decode cost"


def test_reasoning_cost_is_tokens_times_params():
    from bench.reasoning_cost import reasoning_cost

    # Qwen3-0.6B, 1 token, 2 bytes/param, 936 GB/s, $0.25/hr.
    one_token = reasoning_cost(params=600_000_000, tokens=1, bytes_per_param=2,
                               bandwidth=936e9, usd_per_hour=0.25)
    expected = (600e6 * 2 / 936e9) / 3600 * 0.25
    assert one_token == expected

    # 10k-token trace costs 10k x more (same model, same hardware).
    trace = reasoning_cost(params=600_000_000, tokens=10_000, bytes_per_param=2,
                           bandwidth=936e9, usd_per_hour=0.25)
    assert trace == one_token * 10_000


def test_small_model_thinking_longer_costs_same_as_big_model_once():
    """671B x 1 token == 32B x ~21 tokens, to first order."""
    big = 671e9 * 1
    small = 32e9 * (671 / 32)
    assert abs(big - small) < 1e-6

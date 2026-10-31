"""Lecture 41 -- RL infrastructure: rollout (decode) vs learner (training) split."""

from __future__ import annotations

from book.code.rl_loop_profile import step_breakdown
from train.rl_loop import rl_step_time_breakdown


class TestBookArithmetic:
    def test_rollout_dominates_at_long_traces(self):
        short = step_breakdown(prompt_len=256, gen_len=512)
        long = step_breakdown(prompt_len=256, gen_len=8192)
        assert short["rollout_pct"] < 40.0
        assert long["rollout_pct"] > 80.0


class TestImplementation:
    def test_rl_step_time_breakdown_keys_and_scaling(self):
        short = rl_step_time_breakdown(prompt_len=256, gen_len=128, group_size=8, n_prompts=4)
        long = rl_step_time_breakdown(prompt_len=256, gen_len=4096, group_size=8, n_prompts=4)
        assert 0.0 < short["rollout_share"] < long["rollout_share"] < 1.0
        assert long["rollout_s"] == short["rollout_s"] * 32

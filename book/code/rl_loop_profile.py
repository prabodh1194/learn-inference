"""Lecture 41 demo -- Why RL training is 80% an inference problem.

    uv run python book/code/rl_loop_profile.py

Breaks one GRPO step into (1) autoregressive rollout generation (memory-bound
decode, Parts I-V) and (2) forward + backward policy update (compute-bound,
Part VI) across trace lengths.
"""

from __future__ import annotations

ACTIVE_PARAMS = 440_401_920  # Qwen3-0.6B non-embedding params
BW_3090 = 936.2e9            # 936.2 GB/s
FLOPS_3090 = 71e12           # 71 TFLOPS dense tensor core


def step_breakdown(prompt_len: int, gen_len: int, n_prompts: int = 4, group_size: int = 8) -> dict[str, float]:
    total_seqs = n_prompts * group_size
    total_tokens = total_seqs * (prompt_len + gen_len)
    weight_bytes_per_step = ACTIVE_PARAMS * 2
    avg_ctx = prompt_len + gen_len / 2.0
    kv_bytes_per_step = total_seqs * avg_ctx * 114_688
    rollout_s = gen_len * (weight_bytes_per_step + kv_bytes_per_step) / BW_3090
    learner_flops = 8.0 * ACTIVE_PARAMS * total_tokens
    learner_s = learner_flops / (0.50 * FLOPS_3090)
    tot_s = rollout_s + learner_s
    return {
        "rollout_s": rollout_s,
        "learner_s": learner_s,
        "rollout_pct": 100.0 * rollout_s / tot_s,
    }


def print_step_row(p_len: int, g_len: int) -> None:
    res = step_breakdown(p_len, g_len)
    print(
        f"  {p_len:10d}   {g_len:7d}          {res['rollout_s']:7.2f} s           {res['learner_s']:7.2f} s          {res['rollout_pct']:5.1f}%"
    )


def main() -> None:
    print("One GRPO step on RTX 3090 (Qwen3-0.6B, 4 prompts x G=8 = 32 sequences)")
    print("=========================================================================")
    print("  prompt_len   gen_len   rollout (decode)   learner (fwd+bwd)   rollout share")
    print_step_row(256, 512)
    print_step_row(256, 2048)
    print_step_row(256, 4096)
    print_step_row(256, 8192)
    print("\n  Rollout is Parts I-V (decode + KV cache). Learner is Part VI (6ND-8ND compute).")


if __name__ == "__main__":
    main()

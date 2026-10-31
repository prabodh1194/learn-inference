"""L41 -- RL infrastructure: rollout (decode) vs learner (training) time split."""

from __future__ import annotations


def rl_step_time_breakdown(
    prompt_len: int,
    gen_len: int,
    group_size: int,
    n_prompts: int,
    active_params: int = 440_400_000,
    bandwidth: float = 936.2e9,
    peak_flops: float = 71e12,
) -> dict[str, float]:
    """L41. Idealized roofline time for one GRPO step: rollout_s, train_s, rollout_share.

    - Rollout decodes `gen_len` tokens sequentially, reading weights (2 bytes/param)
      once per token step at `bandwidth`: rollout_s = (gen_len * 2 * active_params) / bandwidth.
    - Learner processes all `n_prompts * group_size * (prompt_len + gen_len)` tokens
      in forward + backward (6 * active_params FLOPs/token) at `peak_flops`.
    """
    raise NotImplementedError("L41")

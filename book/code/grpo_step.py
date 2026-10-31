"""Lecture 39 demo -- GRPO group-relative advantages and the Schulman k3 KL estimator.

    uv run python book/code/grpo_step.py

Reproduces the worked 4-completion GRPO advantage calculation from DeepSeekMath
and compares the naive log-ratio KL estimator against the unbiased non-negative
k3 estimator.
"""

from __future__ import annotations

import math


def print_group_advantages() -> None:
    rewards = [1.0, 0.0, 0.8, 0.2]
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    advs = [(r - mean) / std for r in rewards]
    print("GRPO group-relative advantage (G=4 completions for one question)")
    print("=========================================================================")
    print(f"  rewards   : {rewards}")
    print(f"  mean      : {mean:.4f}    population std : {std:.4f}")
    for i, (r, a) in enumerate(zip(rewards, advs, strict=True), 1):
        direction = "REINFORCE (+)" if a > 0 else "SUPPRESS  (-)"
        print(f"  answer {i}  : reward={r:4.1f}  ->  A_hat = ({r:3.1f} - {mean:.1f}) / {std:.4f} = {a:+6.2f}  [{direction}]")
    print()


def print_kl_estimators() -> None:
    print("Per-token KL estimators: naive log(pi/ref) vs Schulman k3 = (ref/pi) - log(ref/pi) - 1")
    print("=========================================================================")
    print("  pi_theta    pi_ref    naive log(pi/ref)    k3 estimator (always >= 0)")
    for pi_p, ref_p in [(0.20, 0.20), (0.40, 0.20), (0.10, 0.20), (0.02, 0.20)]:
        naive = math.log(pi_p / ref_p)
        r = ref_p / pi_p
        k3 = r - math.log(r) - 1.0
        print(f"  {pi_p:8.2f}  {ref_p:8.2f}         {naive:+12.4f}                  {k3:12.4f}")


def main() -> None:
    print_group_advantages()
    print_kl_estimators()


if __name__ == "__main__":
    main()

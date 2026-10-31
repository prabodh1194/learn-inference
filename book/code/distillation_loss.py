"""Lecture 40 demo -- Forward vs Reverse KL and temperature-scaled logit distillation.

    uv run python book/code/distillation_loss.py

Shows why multiplying KL by T^2 preserves gradient magnitude as temperature T
softens the teacher/student distributions, and compares direct RL vs SFT
distillation on a 32B student from the DeepSeek-R1 ablation.
"""

from __future__ import annotations

import math


def softmax(logits: list[float], temp: float) -> list[float]:
    m = max(logits)
    exps = [math.exp((z - m) / temp) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def print_temp_row(z_t: list[float], z_s: list[float], t: float) -> None:
    pt = softmax(z_t, t)
    ps = softmax(z_s, t)
    kl = sum(p * math.log(p / q) for p, q in zip(pt, ps, strict=True))
    grad_unscaled = math.sqrt(sum(((q - p) / t) ** 2 for p, q in zip(pt, ps, strict=True)))
    grad_scaled = (t**2) * grad_unscaled
    print(f"  {t:8.1f}     {kl:10.5f}    {kl*t*t:8.4f}                {grad_unscaled:14.5f}      {grad_scaled:14.5f}")


def print_temp_scaling() -> None:
    z_t = [4.0, 2.0, 0.0, -1.0]
    z_s = [3.0, 2.5, 0.2, -0.5]
    print("Temperature-scaled KL distillation: why the T^2 factor is required")
    print("=========================================================================")
    print("  temp (T)    unscaled KL    T^2 * KL    unscaled grad ||dKL/dz_s||   T^2-scaled grad")
    print_temp_row(z_t, z_s, 1.0)
    print_temp_row(z_t, z_s, 2.0)
    print_temp_row(z_t, z_s, 5.0)
    print_temp_row(z_t, z_s, 10.0)
    print()


def print_r1_ablation() -> None:
    print("DeepSeek-R1 Section 4.2 ablation on Qwen-32B-Base (discovery vs imitation)")
    print("=========================================================================")
    print("  method                                      AIME 2024 pass@1   MATH-500 pass@1")
    print("  Qwen-32B-Base + 10k steps of direct RL          47.0%              83.0%")
    print("  Qwen-32B-Base + SFT distillation from R1        72.6%              94.3%")


def main() -> None:
    print_temp_scaling()
    print_r1_ablation()


if __name__ == "__main__":
    main()

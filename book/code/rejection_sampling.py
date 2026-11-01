from __future__ import annotations
import math


def bon_kl_divergence(n_samples: int) -> float:
    return math.log(n_samples) - (n_samples - 1) / n_samples


def select_top_per_prompt(reward_matrix: list[list[float]]) -> list[tuple[int, int, float]]:
    selected: list[tuple[int, int, float]] = []
    for p_idx, row in enumerate(reward_matrix):
        best_j = max(range(len(row)), key=lambda j: row[j])
        selected.append((p_idx, best_j, row[best_j]))
    return selected


def select_top_k_overall(
    reward_matrix: list[list[float]], k: int
) -> list[tuple[int, int, float]]:
    flat: list[tuple[int, int, float]] = []
    for p_idx, row in enumerate(reward_matrix):
        for c_idx, val in enumerate(row):
            flat.append((p_idx, c_idx, val))
    flat.sort(key=lambda item: item[2], reverse=True)
    return flat[:k]


def _build_bon_lines() -> list[str]:
    lines: list[str] = []
    for n_val in (1, 2, 4, 8, 16, 32, 64):
        kl = math.log(n_val) - (n_val - 1) / n_val
        gauss_gain = math.sqrt(2.0 * math.log(n_val)) if n_val > 1 else 0.0
        lines.append(
            f"  N = {n_val:>2} | KL(BoN || ref) = {kl:6.4f} nats | "
            f"E[max Z] ~ {gauss_gain:5.3f} sigma"
        )
    return lines


def main() -> None:
    print("Best-of-N Analytical KL Budget: KL = ln(N) - (N - 1)/N")
    print("-" * 64)
    print("\n".join(_build_bon_lines()))
    rewards = [
        [0.7, 0.3, 0.5, 0.2],
        [0.4, 0.8, 0.6, 0.5],
        [0.9, 0.3, 0.4, 0.7],
        [0.2, 0.5, 0.8, 0.6],
        [0.5, 0.4, 0.3, 0.6],
    ]
    per_p = select_top_per_prompt(rewards)
    overall = select_top_k_overall(rewards, k=5)
    print("\nRejection Sampling Selection (5 prompts x 4 completions):")
    print(f"  top_per_prompt : {per_p} (mean R = {sum(x[2] for x in per_p)/5:.2f})")
    print(f"  top_5_overall  : {overall} (mean R = {sum(x[2] for x in overall)/5:.2f})")


if __name__ == "__main__":
    main()

from __future__ import annotations


def overopt_gold_reward(d_kl_sqrt: float, alpha: float = 1.0, beta: float = 0.25) -> float:
    return d_kl_sqrt * (alpha - beta * d_kl_sqrt)


def ties_merge_1d(task_vectors: list[list[float]]) -> list[float]:
    n_dims = len(task_vectors[0])
    merged: list[float] = []
    for idx in range(n_dims):
        col = [vec[idx] for vec in task_vectors if vec[idx] != 0.0]
        if not col:
            merged.append(0.0)
            continue
        elected_sign = 1.0 if sum(col) >= 0.0 else -1.0
        agree = [val for val in col if (val > 0.0) == (elected_sign > 0.0)]
        merged.append(sum(agree) / len(agree))
    return merged


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


def _build_overopt_lines() -> list[str]:
    lines: list[str] = []
    for d_val in (0.0, 1.0, 2.0, 3.0, 4.0):
        proxy = 1.0 * d_val
        gold = d_val * (1.0 - 0.25 * d_val)
        lines.append(
            f"  d = sqrt(KL) = {d_val:3.1f} (KL={d_val**2:4.1f}) | "
            f"R_proxy = {proxy:+5.2f} | R_gold = {gold:+5.2f}"
        )
    return lines


def main() -> None:
    print("Gao et al. Over-Optimization Law (alpha=1.0, beta=0.25 -> peak d*=2.0)")
    print("-" * 72)
    print("\n".join(_build_overopt_lines()))
    t_vecs = [[0.6, -0.4, 0.2], [0.4, 0.1, -0.5], [-0.2, -0.6, 0.4]]
    print(f"\nTIES-Merging on 3 task vectors: {ties_merge_1d(t_vecs)}")
    p_at_5 = pass_at_k(10, 8, 5)
    p_pow_5 = (8 / 10) ** 5
    print(f"n=10, c=8 (80% base): pass@5 = {p_at_5:.4f} vs pass^5 = {p_pow_5:.4f}")


if __name__ == "__main__":
    main()

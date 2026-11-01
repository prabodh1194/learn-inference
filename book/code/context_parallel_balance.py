from __future__ import annotations


def naive_ring_causal_blocks(cp_degree: int) -> list[float]:
    return [float(2 * r + 1) for r in range(cp_degree)]


def zigzag_ring_causal_blocks(cp_degree: int) -> list[float]:
    blocks: list[float] = []
    for r in range(cp_degree):
        first_sub = 2 * r + 1
        second_sub = 2 * (2 * cp_degree - 1 - r) + 1
        blocks.append(0.25 * float(first_sub + second_sub))
    return blocks


def ep_all_to_all_mib(
    batch_tokens: int, hidden_dim: int, top_k: int, ep_degree: int
) -> float:
    remote_frac = (ep_degree - 1) / ep_degree
    dispatch_and_combine_bytes = 2 * (2 * batch_tokens * hidden_dim * top_k) * remote_frac
    return dispatch_and_combine_bytes / (1024 ** 2)


def _format_cp_Lines(cp_degree: int) -> list[str]:
    naive = naive_ring_causal_blocks(cp_degree)
    zigzag = zigzag_ring_causal_blocks(cp_degree)
    n_ratio = max(naive) / min(naive)
    z_ratio = max(zigzag) / min(zigzag)
    n_util = (sum(naive) / len(naive)) / max(naive) * 100.0
    z_util = (sum(zigzag) / len(zigzag)) / max(zigzag) * 100.0
    return [
        f"CP={cp_degree} Naive Ring   blocks/rank: {naive} | max/min={n_ratio:.1f}x | util={n_util:.1f}%",
        f"CP={cp_degree} Zig-Zag Ring blocks/rank: {zigzag} | max/min={z_ratio:.1f}x | util={z_util:.1f}%",
    ]


def main() -> None:
    print("Context Parallelism Causal Load Balance (unit = 0.5 * (s/C)^2 pairs)")
    print("-" * 72)
    print("\n".join(_format_cp_Lines(4)))
    print("\n".join(_format_cp_Lines(8)))
    mib_ep8 = ep_all_to_all_mib(4096, 2048, 8, 8)
    print(f"\nMoE EP=8 All-to-All (4096 tok, d=2048, top-8): {mib_ep8:.1f} MiB/layer")


if __name__ == "__main__":
    main()

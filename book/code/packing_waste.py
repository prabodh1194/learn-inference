"""Lecture 32 demo -- sequence packing vs padding waste, and the token-averaging bug.

    uv run python book/code/packing_waste.py

Shows how much compute padded training batches waste on variable-length documents,
how packing recovers it, and how naive micro-batch loss averaging biases gradients.
"""

from __future__ import annotations

import random


def sample_doc_lengths(n: int = 64, max_len: int = 2048, seed: int = 0) -> list[int]:
    rng = random.Random(seed)
    return [min(max_len, max(32, int(rng.expovariate(1 / 420)))) for _ in range(n)]


def padded_cost(lengths: list[int], batch_size: int = 8) -> tuple[int, int]:
    useful = sum(lengths)
    allocated = 0
    for i in range(0, len(lengths), batch_size):
        chunk = lengths[i : i + batch_size]
        allocated += max(chunk) * len(chunk)
    return useful, allocated


def packed_cost(lengths: list[int], bin_len: int = 2048) -> tuple[int, int, int]:
    bins: list[int] = []
    for length in lengths:
        placed = False
        for i, used in enumerate(bins):
            if used + length <= bin_len:
                bins[i] += length
                placed = True
                break
        if not placed:
            bins.append(length)
    return sum(lengths), len(bins) * bin_len, len(bins)


def print_packing_comparison() -> None:
    lengths = sample_doc_lengths()
    u_pad, alloc_pad = padded_cost(lengths, batch_size=8)
    u_pack, alloc_pack, n_bins = packed_cost(lengths, bin_len=2048)
    pad_waste = 1.0 - u_pad / alloc_pad
    pack_waste = 1.0 - u_pack / alloc_pack
    print("64 documents (exponential lengths, mean ~428 tokens, cap 2048)")
    print("=========================================================================")
    print(f"  padded batches (batch=8)   : {u_pad:6,d} useful / {alloc_pad:6,d} allocated  ({pad_waste*100:4.1f}% wasted FLOPs)")
    print(f"  packed into 2048-tok bins  : {u_pack:6,d} useful / {alloc_pack:6,d} allocated  ({pack_waste*100:4.1f}% slack across {n_bins} bins)")
    print(f"  effective throughput gain  : {alloc_pad / alloc_pack:.2f}x fewer padded token-slots\n")


def print_averaging_bug() -> None:
    mb1_tokens, mb1_loss_per_tok = 100, 2.0
    mb2_tokens, mb2_loss_per_tok = 10, 5.0
    naive_mean = 0.5 * (mb1_loss_per_tok + mb2_loss_per_tok)
    true_mean = (mb1_tokens * mb1_loss_per_tok + mb2_tokens * mb2_loss_per_tok) / (mb1_tokens + mb2_tokens)
    print("The gradient-accumulation averaging trap (2 micro-batches)")
    print("=========================================================================")
    print("  micro-batch 1: 100 valid tokens @ loss 2.0/token  (total = 200.0)")
    print("  micro-batch 2:  10 valid tokens @ loss 5.0/token  (total =  50.0)")
    print(f"  naive (L1 + L2) / 2          : {naive_mean:.4f}  (weights each MB2 token 10x heavier!)")
    print(f"  true  sum(loss) / sum(tokens): {true_mean:.4f}")


def main() -> None:
    print_packing_comparison()
    print_averaging_bug()


if __name__ == "__main__":
    main()

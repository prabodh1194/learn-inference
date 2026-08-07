"""Lecture 09 -- why contiguous KV allocation wastes most of your memory.

Arithmetic only, no GPU.

    uv run python book/code/fragmentation.py

Compares reserve-max-per-sequence against block allocation on the same
requests, and reports the metric that matters: how many sequences fit in a
fixed amount of VRAM.
"""

from __future__ import annotations

import random

from book.code.roofline import ModelDims, model_params_bytes

GIB = 1024**3


def contiguous_sequences(vram_bytes: int, d: ModelDims, max_seq_len: int) -> int:
    """Contiguous: every sequence reserves max_seq_len up front.

    You must reserve the worst case, because the tensor can't grow into memory
    that another sequence might take.
    """
    per_seq = d.kv_bytes_per_token() * max_seq_len
    return int(vram_bytes // per_seq)


def paged_sequences(vram_bytes: int, d: ModelDims, actual_lengths: list[int],
                    block_size: int = 16) -> int:
    """Paged: allocate blocks on demand, rounded up to block_size.

    Waste is bounded by (block_size - 1) tokens per sequence -- internal
    fragmentation only, never the unused tail of a reservation.
    """
    per_token = d.kv_bytes_per_token()
    fitted = 0
    used = 0
    for length in actual_lengths:
        blocks = -(-length // block_size)          # ceil
        need = blocks * block_size * per_token
        if used + need > vram_bytes:
            break
        used += need
        fitted += 1
    return fitted


def main() -> None:
    d = ModelDims()
    rng = random.Random(0)

    weights = model_params_bytes(d)
    vram = 24 * GIB                                 # RTX 3090
    kv_budget = vram - weights - 2 * GIB            # leave room for activations

    print(f"RTX 3090: {vram/GIB:.0f} GiB")
    print(f"  weights          {weights/GIB:5.2f} GiB")
    print(f"  activations etc  {2:5.2f} GiB")
    print(f"  -> KV budget     {kv_budget/GIB:5.2f} GiB")
    print(f"  KV per token     {d.kv_bytes_per_token()/1024:.0f} KiB\n")

    # Realistic traffic: most requests short, a few long. Deliberately NOT
    # round numbers -- real token counts don't land on block boundaries, and
    # rounding them would hide the internal fragmentation this demo measures.
    lengths = [rng.randint(64, 4096) for _ in range(2000)]
    mean_len = sum(lengths) / len(lengths)

    print("Traffic: mixed lengths, mean "
          f"{mean_len:.0f} tokens, max 4096\n")

    print("How many concurrent sequences fit?")
    print("=" * 62)
    print(f"{'max_seq_len':>13}{'contiguous':>13}{'paged':>10}{'gain':>9}")
    for max_len in (2048, 4096, 8192, 32768):
        c = contiguous_sequences(kv_budget, d, max_len)
        capped = [min(x, max_len) for x in lengths]
        p = paged_sequences(kv_budget, d, capped)
        gain = f"{p/c:.1f}x" if c else "inf"
        print(f"{max_len:>13}{c:>13}{p:>10}{gain:>9}")

    print("\nContiguous allocation must reserve the WORST CASE for every")
    print("sequence. Support 32k context and a 128-token chat request still")
    print("reserves 32k -- 99.6% of it unused.")

    print("\n\nWhere the memory actually goes (max_seq_len=8192)")
    print("=" * 62)
    max_len = 8192
    n = contiguous_sequences(kv_budget, d, max_len)
    reserved = n * max_len
    actually_used = sum(min(x, max_len) for x in lengths[:n])
    print(f"  sequences fitted     {n}")
    print(f"  tokens reserved      {reserved:,}")
    print(f"  tokens actually used {actually_used:,}")
    print(f"  WASTED               {100*(1-actually_used/reserved):.1f}%")

    print("\n\nBlock size: the internal-fragmentation tradeoff")
    print("=" * 62)
    print("Blocks round up, so each sequence wastes < block_size tokens.")
    print("Smaller blocks waste less memory but need bigger block tables.\n")
    print(f"{'block_size':>12}{'sequences':>12}{'waste/seq':>12}"
          f"{'blocks/seq':>13}")
    for bs in (1, 8, 16, 32, 128, 512):
        p = paged_sequences(kv_budget, d, lengths, block_size=bs)
        avg_waste = sum((-(-x // bs) * bs) - x for x in lengths) / len(lengths)
        avg_blocks = sum(-(-x // bs) for x in lengths) / len(lengths)
        print(f"{bs:>12}{p:>12}{avg_waste:>11.1f}t{avg_blocks:>13.0f}")

    print("\nSmall blocks waste less memory but need longer block tables to")
    print("track (more lookups per attention step). vLLM defaults to 16 --")
    print("waste is already negligible and the table stays short.")

    print("\n\nWhat this buys you")
    print("=" * 62)
    print("More concurrent sequences = bigger decode batches. And from")
    print("Lecture 01, bigger batches directly raise arithmetic intensity on")
    print("a memory-bound phase. Paging is a MEMORY optimization that buys")
    print("you THROUGHPUT.")


if __name__ == "__main__":
    main()

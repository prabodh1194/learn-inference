"""Lecture 07 -- what static batching wastes.

Arithmetic only, no GPU, no model.

    uv run python book/code/batching_waste.py

Simulates a static batch against the same requests served with continuous
batching, and counts the slots that sit idle. The gap between the two is
Lecture 08.
"""

from __future__ import annotations

import random

from bench.workloads import mixed_length, uniform


def _lengths(workload) -> list[int]:
    """Approximate token counts. Word count is close enough for this argument."""
    return [(len(r.prompt.split()), r.max_tokens) for r in workload.requests]


def static_batch_cost(reqs: list[tuple[int, int]], batch_size: int) -> dict:
    """Static batching: a batch runs until its LONGEST member finishes.

    Slots that finish early are dead -- they can't be refilled, because the
    batch is fixed for its lifetime.
    """
    total_slots = useful_slots = 0
    prefill_slots = prefill_useful = 0

    for i in range(0, len(reqs), batch_size):
        chunk = reqs[i:i + batch_size]
        longest_prompt = max(p for p, _ in chunk)
        longest_output = max(o for _, o in chunk)

        # prefill: everything padded to the longest prompt
        prefill_slots += longest_prompt * len(chunk)
        prefill_useful += sum(p for p, _ in chunk)

        # decode: every slot occupied until the longest output completes
        total_slots += longest_output * len(chunk)
        useful_slots += sum(o for _, o in chunk)

    return {
        "decode_slots": total_slots,
        "decode_useful": useful_slots,
        "prefill_slots": prefill_slots,
        "prefill_useful": prefill_useful,
    }


def continuous_batch_cost(reqs: list[tuple[int, int]]) -> dict:
    """Continuous batching: a finished sequence leaves and a new one enters.

    Idealized -- no scheduling overhead, always work available. Real engines
    approach this; they don't reach it.
    """
    return {
        "decode_slots": sum(o for _, o in reqs),
        "decode_useful": sum(o for _, o in reqs),
        "prefill_slots": sum(p for p, _ in reqs),
        "prefill_useful": sum(p for p, _ in reqs),
    }


def report(name: str, reqs: list[tuple[int, int]], batch_size: int) -> None:
    s = static_batch_cost(reqs, batch_size)
    c = continuous_batch_cost(reqs)

    waste = 100 * (1 - s["decode_useful"] / s["decode_slots"])
    pad = 100 * (1 - s["prefill_useful"] / s["prefill_slots"])
    speedup = s["decode_slots"] / c["decode_slots"]

    print(f"\n{name}  (batch={batch_size}, {len(reqs)} requests)")
    print("-" * 58)
    print(f"  prefill padding waste     {pad:5.1f}%")
    print(f"  decode slot waste         {waste:5.1f}%")
    print(f"  static decode steps       {s['decode_slots']:,}")
    print(f"  continuous decode steps   {c['decode_slots']:,}")
    print(f"  -> continuous batching is {speedup:.2f}x better here")


def main() -> None:
    random.seed(0)

    print("Static batching: fix the batch, run until the LONGEST member ends.")
    print("Short requests finish early, but their slots stay occupied.")

    # Uniform: everything the same length. Static batching looks great --
    # which is exactly why benchmarking on uniform load is misleading.
    u = _lengths(uniform(n=32, prompt_words=64, max_tokens=128))
    report("uniform (all requests identical)", u, batch_size=8)

    # Mixed: the real world.
    m = _lengths(mixed_length(n=32, seed=0))
    report("mixed_length (realistic)", m, batch_size=8)

    print("\n\nWaste vs. batch size, on mixed load")
    print("=" * 58)
    print(f"{'batch':>7}{'decode waste':>15}{'vs continuous':>16}")
    for bs in (1, 2, 4, 8, 16, 32):
        s = static_batch_cost(m, bs)
        c = continuous_batch_cost(m)
        waste = 100 * (1 - s["decode_useful"] / s["decode_slots"])
        print(f"{bs:>7}{waste:>14.1f}%{s['decode_slots']/c['decode_slots']:>15.2f}x")

    print("\nBigger batches -> better GPU utilization AND more waste.")
    print("Static batching cannot have both. That is the whole problem.")

    print("\n\nWhy uniform benchmarks lie")
    print("=" * 58)
    print("On uniform load, static batching wastes almost nothing, so it looks")
    print("as good as continuous batching. Benchmark only on uniform load and")
    print("you will conclude the scheduler in Lecture 08 was pointless.")
    print("\nMatch the workload to the optimization, or you cannot see it.")


if __name__ == "__main__":
    main()

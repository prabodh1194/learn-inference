"""M2.1 (Lecture 15) -- profiling, and the arithmetic that keeps it honest.

The ranking helpers below run anywhere and are implemented, because they are
bookkeeping rather than the lesson. `profile_decode` needs CUDA and is yours
to write.

The function worth internalizing is `amdahl_speedup`. Before optimizing any
kernel, compute the ceiling: a kernel at 8% of runtime made 3x faster buys you
5.6% end to end. If that's not worth your weekend, you've just saved a weekend.
"""

from __future__ import annotations


def rank_kernels(times: dict[str, float]) -> list[tuple[str, float, float]]:
    """(name, time, share_of_total), sorted slowest first.

    This ordering is your work queue for Lectures 16-20. Optimize the top of
    the list, not the interesting part of it.
    """
    total = sum(times.values())
    if total <= 0:
        return []
    return sorted(
        ((name, t, t / total) for name, t in times.items()),
        key=lambda row: row[1],
        reverse=True,
    )


def amdahl_speedup(share: float, speedup: float) -> float:
    """End-to-end speedup from making `share` of runtime `speedup` times faster.

        amdahl_speedup(0.08, 3.0)  -> 1.056   (8% of runtime, 3x faster: +5.6%)
        amdahl_speedup(0.60, 2.0)  -> 1.429   (60% of runtime, 2x faster: +43%)
        amdahl_speedup(0.10, inf)  -> 1.111   (even removing it entirely)

    The last case is the useful one: an infinitely fast kernel only removes
    its own share. That is the hard ceiling on any single optimization.
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must be in [0, 1], got {share}")
    if speedup <= 0:
        raise ValueError(f"speedup must be positive, got {speedup}")
    remaining = share / speedup if speedup != float("inf") else 0.0
    return 1.0 / ((1.0 - share) + remaining)


def bandwidth_utilization(
    bytes_moved: float, seconds: float, peak_bandwidth: float
) -> float:
    """Achieved bandwidth as a fraction of peak.

    Lecture 15's stop condition. Decode is memory-bound (Lecture 02), so a
    good decode kernel should reach 0.7-0.9. At 0.85 there is little left to
    win and you should look elsewhere; at 0.3 there is real headroom.
    """
    if seconds <= 0 or peak_bandwidth <= 0:
        return float("nan")
    return (bytes_moved / seconds) / peak_bandwidth


def report(times: dict[str, float], peak_bandwidth: float | None = None) -> None:
    """Print the ranking table that Lectures 16-20 must cite."""
    ranked = rank_kernels(times)
    print(f"\n{'kernel':<28}{'time':>12}{'share':>9}{'3x buys':>10}")
    print("-" * 59)
    for name, t, share in ranked:
        gain = (amdahl_speedup(share, 3.0) - 1) * 100
        print(f"{name:<28}{t:>10.2f}ms{share:>8.1%}{gain:>9.1f}%")
    print("\n'3x buys' is the end-to-end gain if that kernel got 3x faster.")
    print("Optimize the top row, and only if the last column justifies it.")


def profile_decode(model, tokenizer, batch_size: int = 1, steps: int = 16):
    """M2.1. Profile a steady-state decode loop; return rank_kernels() output.

    Use torch.profiler with CPU and CUDA activities. Discard warmup steps --
    the first iterations include context setup and autotuning, and profiling
    a cold start tells you about startup, not serving.

    Compare Self CPU against Self CUDA first: if CPU dominates, you are
    launch-bound and belong in Lecture 13, not here.
    """
    raise NotImplementedError("M2.1")

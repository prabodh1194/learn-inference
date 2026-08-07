"""Plots. The point of each is to make a bottleneck *visible*.

  time_per_token_curve  M0.3  the quadratic pain -- climbs without a KV cache,
                              flattens with one. Your first "aha".
  roofline              M0.4  arithmetic intensity vs. hardware ops:byte;
                              shows prefill and decode landing on opposite sides
  latency_throughput    M4.2  the knee: where throughput stalls and p99 explodes
  scaling               M3.3  TP scaling vs. ideal linear; shows comms cost
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

RESULTS_DIR = Path(__file__).parent / "results"


def _plt():
    import matplotlib

    matplotlib.use("Agg")  # headless: works over SSH on a rented box
    import matplotlib.pyplot as plt

    return plt


def _save(fig, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"saved -> {path}")
    return path


def time_per_token_curve(
    series: dict[str, tuple[Sequence[int], Sequence[float]]],
    name: str = "m03-time-per-token.png",
) -> Path:
    """M0.3 / M1.1. series: label -> (positions, seconds_per_token).

    Without a KV cache each step re-attends over all prior tokens, so per-token
    time climbs with position. With one, it should be roughly flat.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (xs, ys) in series.items():
        ax.plot(xs, [y * 1000 for y in ys], marker="o", ms=3, label=label)
    ax.set_xlabel("token position in sequence")
    ax.set_ylabel("time per token (ms)")
    ax.set_title("Per-token latency vs. sequence position")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, name)


def roofline(
    peak_flops: float,
    peak_bandwidth: float,
    points: dict[str, tuple[float, float]],
    name: str = "m04-roofline.png",
    device_label: str = "",
) -> Path:
    """M0.4 / M2.3. Book §2.4.

    peak_flops      FLOP/s   peak_bandwidth  bytes/s
    points          label -> (arithmetic_intensity_ops_per_byte, achieved_flops)

    The ridge at peak_flops/peak_bandwidth is the ops:byte ratio. Left of it =
    memory-bound (decode), right = compute-bound (prefill).
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ridge = peak_flops / peak_bandwidth
    xs = [ridge * (2**i) for i in range(-12, 8)]
    ax.plot(xs, [min(peak_bandwidth * x, peak_flops) for x in xs],
            "k-", lw=2, label="roofline")
    ax.axvline(ridge, ls="--", c="gray", alpha=0.7)
    ax.text(ridge, peak_flops * 0.12, f"  ops:byte = {ridge:.0f}",
            rotation=90, va="bottom", fontsize=8, color="gray")

    for label, (intensity, achieved) in points.items():
        ax.plot([intensity], [achieved], "o", ms=9, label=label)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved FLOP/s")
    ax.set_title(f"Roofline {device_label}".strip())
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, name)


def latency_throughput(
    points: Sequence[tuple[float, float, float]],
    name: str = "m42-latency-throughput.png",
    label: str = "",
) -> Path:
    """M4.2. points: (throughput_tok_s, p50_latency_s, p99_latency_s).

    Sweep concurrency and find the knee -- the point past which throughput
    stops improving but p99 runs away. That knee is your operating limit.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    tps = [p[0] for p in points]
    ax.plot(tps, [p[1] for p in points], marker="o", label=f"p50 {label}".strip())
    ax.plot(tps, [p[2] for p in points], marker="s", ls="--",
            label=f"p99 {label}".strip())
    ax.set_xlabel("output throughput (tok/s)")
    ax.set_ylabel("latency (s)")
    ax.set_title("Latency vs. throughput (sweep concurrency)")
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, name)


def scaling(
    n_devices: Sequence[int],
    throughput: Sequence[float],
    name: str = "m33-scaling.png",
) -> Path:
    """M3.3. Measured vs. ideal-linear scaling. The gap is communication cost."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    base = throughput[0]
    ax.plot(n_devices, throughput, marker="o", label="measured")
    ax.plot(n_devices, [base * n / n_devices[0] for n in n_devices],
            ls="--", c="gray", label="ideal linear")
    ax.set_xlabel("devices")
    ax.set_ylabel("throughput (tok/s)")
    ax.set_title("Tensor-parallel scaling")
    ax.set_xticks(list(n_devices))
    ax.grid(alpha=0.3)
    ax.legend()
    return _save(fig, name)

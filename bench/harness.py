"""Measurement core. Built before any engine code, on purpose.

Every milestone in the plan is gated on a number this module produces. If a
change doesn't move a number in here, you don't yet know whether it worked.

Metric definitions (Inference Engineering §1.4):
  TTFT  time to first token       -- dominated by prefill, compute-bound
  TPOT  time per output token     -- dominated by decode, memory-bound
  tok/s output tokens per second  -- 1/TPOT for a single stream
  p50/p90/p99                     -- report these, never the mean alone;
                                     tail latency is what users actually feel
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

RESULTS_DIR = Path(__file__).parent / "results"


# --------------------------------------------------------------------------
# device / sync
# --------------------------------------------------------------------------


def _torch():
    import torch

    return torch


def detect_device() -> str:
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str | None = None) -> None:
    """Block until queued GPU work completes.

    Without this, timings on CUDA/MPS measure how fast you enqueued kernels,
    not how fast they ran. This single call is the difference between real
    numbers and nonsense.
    """
    torch = _torch()
    device = device or detect_device()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def peak_memory_bytes(device: str | None = None) -> int:
    torch = _torch()
    device = device or detect_device()
    if device == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device == "mps":
        return int(torch.mps.current_allocated_memory())
    return 0


def reset_peak_memory(device: str | None = None) -> None:
    torch = _torch()
    device = device or detect_device()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()


@contextmanager
def timer(device: str | None = None) -> Iterator[Callable[[], float]]:
    """Wall-clock timer that synchronizes on both ends. Yields elapsed-seconds getter."""
    synchronize(device)
    start = time.perf_counter()
    elapsed = float("nan")

    def get() -> float:
        return elapsed

    try:
        yield get
    finally:
        synchronize(device)
        elapsed = time.perf_counter() - start


# --------------------------------------------------------------------------
# percentiles
# --------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile. q in [0, 100]."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


@dataclass
class Distribution:
    """Summary of a set of measurements. Always carries the tail."""

    count: int
    mean: float
    p50: float
    p90: float
    p99: float
    min: float
    max: float

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "Distribution":
        vals = [v for v in values if v == v]  # drop NaN
        if not vals:
            return cls(0, *([float("nan")] * 6))
        return cls(
            count=len(vals),
            mean=statistics.fmean(vals),
            p50=percentile(vals, 50),
            p90=percentile(vals, 90),
            p99=percentile(vals, 99),
            min=min(vals),
            max=max(vals),
        )


# --------------------------------------------------------------------------
# per-request record
# --------------------------------------------------------------------------


@dataclass
class RequestRecord:
    """One request's timeline.

    Call `mark_token()` on every generated token; the first call is TTFT.
    """

    prompt_tokens: int = 0
    output_tokens: int = 0
    arrival: float = field(default_factory=time.perf_counter)
    start: float | None = None
    first_token: float | None = None
    end: float | None = None
    token_times: list[float] = field(default_factory=list)

    def mark_start(self) -> None:
        self.start = time.perf_counter()

    def mark_token(self) -> None:
        now = time.perf_counter()
        if self.first_token is None:
            self.first_token = now
        self.token_times.append(now)
        self.output_tokens = len(self.token_times)

    def mark_end(self) -> None:
        self.end = time.perf_counter()

    # -- derived ----------------------------------------------------------

    @property
    def ttft(self) -> float:
        """Time to first token, measured from arrival (includes queue wait)."""
        if self.first_token is None:
            return float("nan")
        return self.first_token - self.arrival

    @property
    def queue_time(self) -> float:
        if self.start is None:
            return float("nan")
        return self.start - self.arrival

    @property
    def tpot(self) -> float:
        """Mean time per output token, excluding the first (that's TTFT's job)."""
        if len(self.token_times) < 2:
            return float("nan")
        span = self.token_times[-1] - self.token_times[0]
        return span / (len(self.token_times) - 1)

    @property
    def inter_token_latencies(self) -> list[float]:
        t = self.token_times
        return [t[i] - t[i - 1] for i in range(1, len(t))]

    @property
    def latency(self) -> float:
        if self.end is None:
            return float("nan")
        return self.end - self.arrival

    @property
    def output_tps(self) -> float:
        tp = self.tpot
        return 1.0 / tp if tp and tp == tp and tp > 0 else float("nan")


# --------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------


@dataclass
class BenchResult:
    """Aggregate of a run. Serialized to bench/results/ as the milestone record."""

    name: str
    milestone: str
    config: dict[str, Any]
    n_requests: int
    wall_time: float
    prompt_tokens: int
    output_tokens: int
    ttft: Distribution
    tpot: Distribution
    latency: Distribution
    itl: Distribution
    peak_memory_bytes: int
    device: str
    env: dict[str, Any]

    @property
    def output_throughput(self) -> float:
        """Output tokens/sec across all requests -- the throughput number."""
        return self.output_tokens / self.wall_time if self.wall_time else float("nan")

    @property
    def request_throughput(self) -> float:
        return self.n_requests / self.wall_time if self.wall_time else float("nan")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["output_throughput"] = self.output_throughput
        d["request_throughput"] = self.request_throughput
        return d

    def save(self, path: Path | str | None = None) -> Path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        if path is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            path = RESULTS_DIR / f"{self.milestone}-{self.name}-{stamp}.json"
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    def summary(self) -> str:
        mb = self.peak_memory_bytes / 1024**2
        return "\n".join(
            [
                f"{self.milestone}  {self.name}  [{self.device}]",
                f"  requests      {self.n_requests}   wall {self.wall_time:.2f}s",
                f"  tokens        {self.prompt_tokens} prompt / {self.output_tokens} output",
                f"  throughput    {self.output_throughput:.1f} tok/s   "
                f"{self.request_throughput:.2f} req/s",
                f"  TTFT    p50 {self.ttft.p50 * 1000:.1f}ms  "
                f"p90 {self.ttft.p90 * 1000:.1f}ms  p99 {self.ttft.p99 * 1000:.1f}ms",
                f"  TPOT    p50 {self.tpot.p50 * 1000:.2f}ms  "
                f"p90 {self.tpot.p90 * 1000:.2f}ms  p99 {self.tpot.p99 * 1000:.2f}ms",
                f"  latency p50 {self.latency.p50:.2f}s  "
                f"p90 {self.latency.p90:.2f}s  p99 {self.latency.p99:.2f}s",
                f"  peak mem      {mb:.1f} MiB",
            ]
        )


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        torch = _torch()
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info


def summarize(
    records: Sequence[RequestRecord],
    *,
    name: str,
    milestone: str,
    wall_time: float,
    config: dict[str, Any] | None = None,
    device: str | None = None,
) -> BenchResult:
    """Collapse per-request records into a BenchResult."""
    device = device or detect_device()
    itl: list[float] = []
    for r in records:
        itl.extend(r.inter_token_latencies)

    return BenchResult(
        name=name,
        milestone=milestone,
        config=config or {},
        n_requests=len(records),
        wall_time=wall_time,
        prompt_tokens=sum(r.prompt_tokens for r in records),
        output_tokens=sum(r.output_tokens for r in records),
        ttft=Distribution.from_values([r.ttft for r in records]),
        tpot=Distribution.from_values([r.tpot for r in records]),
        latency=Distribution.from_values([r.latency for r in records]),
        itl=Distribution.from_values(itl),
        peak_memory_bytes=peak_memory_bytes(device),
        device=device,
        env=environment(),
    )


@contextmanager
def benchmark(
    name: str, milestone: str, *, config: dict[str, Any] | None = None
) -> Iterator[list[RequestRecord]]:
    """Collect records, then print + save a BenchResult.

        with benchmark("naive-greedy", "M0.1") as records:
            for prompt in prompts:
                rec = RequestRecord(prompt_tokens=len(prompt))
                records.append(rec)
                ...  # rec.mark_start() / mark_token() / mark_end()
    """
    device = detect_device()
    reset_peak_memory(device)
    records: list[RequestRecord] = []
    synchronize(device)
    t0 = time.perf_counter()
    try:
        yield records
    finally:
        synchronize(device)
        wall = time.perf_counter() - t0
        result = summarize(
            records, name=name, milestone=milestone, wall_time=wall,
            config=config, device=device,
        )
        print(result.summary())
        print(f"  saved -> {result.save()}")

"""Shared plumbing for the lecture benchmarks.

The bench scripts differ only in which engine path they call and which
workload they call it on. That common shape lives here so each script stays
short enough to read in one screen.

Not a lecture demo itself -- nothing to run directly.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Iterable

from bench.harness import BenchResult, RequestRecord, detect_device, summarize
from bench.workloads import Request, Workload


def load_model_or_exit():
    """Load Qwen3-0.6B, or exit with the lecture that unblocks you."""
    try:
        from engine.model import load
    except ImportError as exc:
        sys.exit(f"cannot import engine.model: {exc}")
    try:
        return load()
    except NotImplementedError:
        sys.exit("Implement engine/model.py::load first (Lecture 03).")
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"could not load the model: {exc}\nTry: uv run python scripts/fetch_model.py")


def require(fn_name: str, module: str = "engine.generate"):
    """Import a generation function, or explain which lecture provides it."""
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        sys.exit(f"cannot import {module}: {exc}")
    fn = getattr(mod, fn_name, None)
    if fn is None:
        sys.exit(f"{module}.{fn_name} does not exist yet.")
    return fn


def run_sequential(
    generate: Callable,
    model,
    tokenizer,
    requests: Iterable[Request],
    *,
    name: str,
    milestone: str,
    config: dict | None = None,
) -> BenchResult:
    """One request at a time. The baseline every batched path is compared to."""
    records: list[RequestRecord] = []
    t0 = time.perf_counter()
    for req in requests:
        rec = RequestRecord(prompt_tokens=len(tokenizer(req.prompt).input_ids))
        records.append(rec)
        rec.mark_start()
        try:
            generate(model, tokenizer, req.prompt,
                     max_tokens=req.max_tokens, on_token=rec.mark_token)
        except NotImplementedError as exc:
            sys.exit(f"not implemented yet: {exc}")
        rec.mark_end()
    wall = time.perf_counter() - t0
    return summarize(records, name=name, milestone=milestone,
                     wall_time=wall, config=config)


def compare(results: dict[str, BenchResult], *, baseline: str) -> None:
    """Print a before/after table. The point of every bench script."""
    base = results[baseline]
    print(f"\n{'':22}{'tok/s':>10}{'TTFT p50':>11}{'TTFT p99':>11}"
          f"{'lat p99':>10}{'vs base':>10}")
    print("-" * 74)
    for label, r in results.items():
        speedup = r.output_throughput / base.output_throughput if base.output_throughput else float("nan")
        print(f"{label:<22}{r.output_throughput:>10.1f}"
              f"{r.ttft.p50 * 1000:>10.1f}ms{r.ttft.p99 * 1000:>10.1f}ms"
              f"{r.latency.p99:>9.2f}s{speedup:>9.2f}x")
    print()
    for r in results.values():
        r.save()


def banner(title: str, workload: Workload | None = None) -> None:
    print(f"\n{title}")
    print("=" * max(len(title), 60))
    print(f"device: {detect_device()}")
    if workload is not None:
        print(f"workload: {workload.name} ({len(workload)} requests)")
        if workload.note:
            print(f"  {workload.note}")
    print()

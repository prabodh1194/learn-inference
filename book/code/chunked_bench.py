"""Lecture 11 -- chunked prefill: judge it on p99, not the mean.

    uv run python book/code/chunked_bench.py

Uses long_prefill: 8 long prompts (~3000 tokens) mixed with 24 short ones.
Long prefills monopolize steps and stall everyone else, so the number that
matters is p99 latency of the SHORT requests (tagged in the workload).

PREDICT what happens to mean, p99, and throughput before running.
"""

from __future__ import annotations

import argparse
import time

from bench.harness import RequestRecord, summarize
from bench.workloads import long_prefill
from book.code._runner import banner, load_model_or_exit, require


def run(generate, model, tok, workload, chunk_size, label):
    """Tag records so short and long requests can be reported separately."""
    short, long_, t0 = [], [], time.perf_counter()
    for req in workload:
        rec = RequestRecord(prompt_tokens=len(tok(req.prompt).input_ids))
        (short if req.tag == "short" else long_).append(rec)
        rec.mark_start()
        generate(model, tok, req.prompt, max_tokens=req.max_tokens,
                 on_token=rec.mark_token)
        rec.mark_end()
    wall = time.perf_counter() - t0
    cfg = {"chunk_size": chunk_size}
    return (
        summarize(short, name=f"{label}-short", milestone="M1.7",
                  wall_time=wall, config=cfg),
        summarize(long_, name=f"{label}-long", milestone="M1.7",
                  wall_time=wall, config=cfg),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[0, 128, 512, 2048])
    args = ap.parse_args()

    model, tok = load_model_or_exit()
    generate = require("generate_cached")
    workload = long_prefill(n=4, n_short=12)
    banner("Chunked prefill -- p99 of the SHORT requests", workload)

    print(f"{'chunk':>8}{'short p50':>12}{'short p99':>12}"
          f"{'long p99':>12}{'tok/s':>10}")
    print("-" * 54)
    for cs in args.chunk_sizes:
        # chunk 0 = disabled, i.e. the Lecture 08 baseline
        s, l = run(generate, model, tok, workload, cs,
                   f"chunk{cs or 'off'}")
        label = "off" if cs == 0 else str(cs)
        print(f"{label:>8}{s.latency.p50:>11.2f}s{s.latency.p99:>11.2f}s"
              f"{l.latency.p99:>11.2f}s{s.output_throughput:>10.1f}")
        s.save(); l.save()

    print("\nExpect: p99 of the short requests improves a lot, the MEAN barely")
    print("moves, and throughput is roughly flat. Watching only the mean would")
    print("tell you this lecture did nothing.")


if __name__ == "__main__":
    main()

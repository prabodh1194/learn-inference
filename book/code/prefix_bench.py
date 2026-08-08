"""Lecture 10 -- prefix caching, and why prompt ORDER decides the win.

    uv run python book/code/prefix_bench.py

Runs shared_prefix against late_divergence. Near-identical token counts,
opposite cache behaviour: one shares a 400-token system prompt, the other
puts novel content first and reuses nothing.

PREDICT the TTFT for each before running.
"""

from __future__ import annotations

import argparse

from bench.workloads import late_divergence, shared_prefix
from book.code._runner import banner, compare, load_model_or_exit, require, run_sequential


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    model, tok = load_model_or_exit()
    cached = require("generate_cached")
    paged = require("generate_paged")

    for workload in (shared_prefix(n=args.n), late_divergence(n=args.n)):
        banner(f"Prefix caching -- {workload.name}", workload)
        toks = [len(tok(r.prompt).input_ids) for r in workload]
        print(f"prompt tokens: min {min(toks)}, max {max(toks)}\n")

        results = {
            "no prefix cache": run_sequential(
                cached, model, tok, workload,
                name=f"nocache-{workload.name}", milestone="M1.6",
                config={"prefix_caching": False, "workload": workload.name}),
            "prefix cache": run_sequential(
                paged, model, tok, workload,
                name=f"prefix-{workload.name}", milestone="M1.6",
                config={"prefix_caching": True, "workload": workload.name}),
        }
        compare(results, baseline="no prefix cache")

    print("Expect: TTFT collapses on shared_prefix after the first request,")
    print("and barely moves on late_divergence. Same tokens, different order.")
    print("\nReport your cache hit rate for both -- that's the real metric.")


if __name__ == "__main__":
    main()

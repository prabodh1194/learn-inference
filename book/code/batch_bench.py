"""Lecture 07 -- static batching: the gain, and the waste.

    uv run python book/code/batch_bench.py

Runs uniform and mixed_length. The contrast is the lesson: batching looks
great on uniform load and mediocre on realistic load, and it's the SAME code.
"""

from __future__ import annotations

import argparse
import sys
import time

from bench.harness import RequestRecord, summarize
from bench.workloads import mixed_length, uniform
from book.code._runner import banner, compare, load_model_or_exit, require, run_sequential


def run_batched(generate, model, tok, workload, batch_size: int, label: str):
    """Static batching: fixed groups, each running until its longest member ends."""
    reqs = list(workload)
    records, t0 = [], time.perf_counter()

    for i in range(0, len(reqs), batch_size):
        chunk = reqs[i:i + batch_size]
        # A static batch shares one max_tokens -- the largest in the group.
        # That IS the waste: short requests hold their slots to the end.
        longest = max(r.max_tokens for r in chunk)
        recs = [RequestRecord(prompt_tokens=len(tok(r.prompt).input_ids)) for r in chunk]
        for rec in recs:
            rec.mark_start()
        records.extend(recs)

        def on_token(_recs=recs):
            for rec in _recs:          # one step advances every slot
                rec.mark_token()

        try:
            generate(model, tok, [r.prompt for r in chunk],
                     max_tokens=longest, on_token=on_token)
        except NotImplementedError as exc:
            sys.exit(f"not implemented yet: {exc}")
        for rec in recs:
            rec.mark_end()

    wall = time.perf_counter() - t0
    return summarize(records, name=label, milestone="M1.3", wall_time=wall,
                     config={"batch_size": batch_size, "workload": workload.name})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    model, tok = load_model_or_exit()
    gen_one = require("generate_cached")
    gen_batch = require("generate_batched")

    for workload in (uniform(n=args.n), mixed_length(n=args.n, seed=0)):
        banner(f"Static batching -- {workload.name}", workload)
        results = {"sequential": run_sequential(
            gen_one, model, tok, workload,
            name=f"seq-{workload.name}", milestone="M1.3",
            config={"workload": workload.name})}
        for bs in args.batch_sizes:
            if bs == 1:
                continue
            results[f"batch={bs}"] = run_batched(
                gen_batch, model, tok, workload, bs, f"batch{bs}-{workload.name}")
        compare(results, baseline="sequential")

    print("Expect: a large gain on uniform, a much smaller one on mixed_length.")
    print("That gap is the padding and slot waste Lecture 08 eliminates.")


if __name__ == "__main__":
    main()

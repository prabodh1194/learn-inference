"""Lecture 03 -- measure the quadratic curve.

Needs the model and your generate_naive implementation.

    uv run python book/code/naive_bench.py

WRITE YOUR PREDICTION DOWN FIRST (notes/00-baseline/README.md). Per-token
latency vs. position: flat, linear, or something else? Commit to an answer
before you see the plot -- that is where the intuition actually comes from.

After Lecture 05, run with --cached to overlay the KV-cache version.
"""

from __future__ import annotations

import argparse
import sys

from bench.harness import RequestRecord, benchmark, detect_device
from bench.plot import time_per_token_curve

PROMPT = "The KV cache exists because"
LENGTHS = (128, 256, 512, 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true",
                    help="use generate_cached (Lecture 05) instead")
    ap.add_argument("--lengths", type=int, nargs="+", default=list(LENGTHS))
    args = ap.parse_args()

    try:
        from engine.model import load
    except ImportError:
        sys.exit("engine/model.py not importable")

    if args.cached:
        from engine.generate import generate_cached as generate
        label, milestone = "cached", "M1.1"
    else:
        from engine.generate import generate_naive as generate
        label, milestone = "naive", "M0.3"

    try:
        model, tok = load()
    except NotImplementedError:
        sys.exit("Implement engine/model.py::load first (Lecture 03).")

    print(f"device: {detect_device()}   mode: {label}\n")

    series = {}
    for n in args.lengths:
        with benchmark(f"{label}-{n}", milestone,
                       config={"max_tokens": n, "cached": args.cached}) as recs:
            rec = RequestRecord(prompt_tokens=len(tok(PROMPT).input_ids))
            recs.append(rec)
            rec.mark_start()
            try:
                generate(model, tok, PROMPT, max_tokens=n, on_token=rec.mark_token)
            except NotImplementedError:
                sys.exit(f"Implement generate_{label} first.")
            rec.mark_end()

        itl = rec.inter_token_latencies
        if itl:
            series[f"{label} n={n}"] = (list(range(1, len(itl) + 1)), itl)
            # The headline: is the LAST token slower than the FIRST?
            head = sum(itl[:20]) / min(20, len(itl))
            tail = sum(itl[-20:]) / min(20, len(itl))
            print(f"  first 20 tokens: {head*1000:6.1f} ms/tok")
            print(f"  last  20 tokens: {tail*1000:6.1f} ms/tok"
                  f"   ({tail/head:.2f}x slower)\n")

    if series:
        name = f"l03-time-per-token{'-cached' if args.cached else ''}.png"
        time_per_token_curve(series, name=name)
        print("\nWas your prediction right? Record it -- including if it wasn't.")


if __name__ == "__main__":
    main()

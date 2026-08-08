"""Lecture 12 -- speculative decoding, and why the workload decides.

    uv run python book/code/spec_bench.py

Runs code_completion (repetitive; n-grams hit) against prose (novel; they
miss). Report ACCEPTANCE RATE alongside tok/s -- without it you can't tell a
real win from a lucky one.
"""

from __future__ import annotations

import argparse

from bench.workloads import code_completion, prose
from book.code._runner import banner, compare, load_model_or_exit, require, run_sequential


def acceptance_rate(workload, n_draft: int) -> float:
    """Offline estimate: how often would the prompt's own n-grams fire?

    Not the true acceptance rate (that needs the model to agree), but it
    isolates the WORKLOAD's predictability, which is the lecture's point.
    """
    from engine.speculative import NgramSpeculator

    spec = NgramSpeculator(n=3, max_draft_tokens=n_draft)
    hits = total = 0
    for req in workload:
        toks = [abs(hash(w)) % 50000 for w in req.prompt.split()]
        for i in range(4, len(toks)):
            total += 1
            if spec.propose(toks[:i]):
                hits += 1
    return hits / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-lengths", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    model, tok = load_model_or_exit()
    baseline = require("generate_cached")
    spec_gen = require("generate_speculative")

    for workload in (code_completion(n=args.n), prose(n=args.n)):
        banner(f"Speculative decoding -- {workload.name}", workload)
        try:
            print(f"n-gram hit rate (workload predictability): "
                  f"{acceptance_rate(workload, 8):.1%}\n")
        except NotImplementedError:
            print("(implement NgramSpeculator to see the hit rate)\n")

        results = {"no speculation": run_sequential(
            baseline, model, tok, workload,
            name=f"base-{workload.name}", milestone="M1.8",
            config={"speculative": False})}

        for nd in args.draft_lengths:
            def gen(m, t, p, max_tokens, on_token, _nd=nd):
                return spec_gen(m, t, p, max_tokens=max_tokens,
                                on_token=on_token, n_draft=_nd)
            results[f"draft={nd}"] = run_sequential(
                gen, model, tok, workload,
                name=f"spec{nd}-{workload.name}", milestone="M1.8",
                config={"speculative": True, "n_draft": nd})

        compare(results, baseline="no speculation")

    print("Expect: solid gains on code_completion, little or nothing on prose,")
    print("and a PEAK in the draft-length sweep -- past it you pay to verify")
    print("tokens that get thrown away. Compare your peak to the field notes' 5.")


if __name__ == "__main__":
    main()

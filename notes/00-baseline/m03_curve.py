"""M0.3 -- plot per-token latency vs. position. The quadratic pain.

    uv run python notes/00-baseline/m03_curve.py

PREDICT BEFORE RUNNING: step n attends over n prior tokens, so per-token time
should climb roughly linearly. Write your guess in the note first.

After M1.1, re-run with the cached path and overlay the two -- the flattening
is the payoff.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bench.harness import RequestRecord  # noqa: E402
from bench.plot import time_per_token_curve  # noqa: E402


def main() -> None:
    from engine.generate import generate_naive
    from engine.model import load

    model, tok = load()
    prompt = "The KV cache exists because"

    series = {}
    for n in (128, 256, 512, 1024):
        rec = RequestRecord(prompt_tokens=len(tok(prompt).input_ids))
        rec.mark_start()
        generate_naive(model, tok, prompt, max_tokens=n, on_token=rec.mark_token)
        rec.mark_end()
        itl = rec.inter_token_latencies
        series[f"naive n={n}"] = (list(range(1, len(itl) + 1)), itl)
        print(f"n={n:<5} tok/s={rec.output_tps:6.1f}  "
              f"last-token={itl[-1]*1000:.1f}ms" if itl else f"n={n}")

    time_per_token_curve(series)


if __name__ == "__main__":
    main()

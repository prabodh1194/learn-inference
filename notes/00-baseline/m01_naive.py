"""M0.1 -- naive generation, no KV cache. Deliberately slow.

    uv run python notes/00-baseline/m01_naive.py

Implement engine/model.py::load and engine/generate.py::generate_naive first.
Expect this to be slow -- that is the entire point. Feel the pain before fixing it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bench.harness import RequestRecord, benchmark  # noqa: E402


def main() -> None:
    from engine.generate import generate_naive
    from engine.model import load

    model, tok = load()
    prompts = [
        "Explain why decode is memory-bound.",
        "def paged_attention(q, k_cache, v_cache, block_table):",
        "The KV cache exists because",
    ]

    with benchmark("naive-greedy", "M0.1", config={"cache": False}) as records:
        for p in prompts:
            rec = RequestRecord(prompt_tokens=len(tok(p).input_ids))
            records.append(rec)
            rec.mark_start()
            generate_naive(model, tok, p, max_tokens=64, on_token=rec.mark_token)
            rec.mark_end()


if __name__ == "__main__":
    main()

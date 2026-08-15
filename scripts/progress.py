"""Where am I in the course?

    uv run python scripts/progress.py

Runs the test suite and reports which lectures are built. Failing tests aren't
breakage -- they're the lectures you haven't done yet.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

LECTURES = [
    ("02", "Arithmetic intensity", "test_02_roofline.py"),
    ("03", "Naive generation", "test_03_generation.py"),
    ("04", "Measuring", "test_04_measuring.py"),
    ("05", "KV cache", "test_05_kv_cache.py"),
    ("06", "Sampling", "test_06_sampling.py"),
    ("06b", "Search decoding", "test_06b_beam.py"),
    ("07", "Static batching", "test_07_batching.py"),
    ("08", "Continuous batching", "test_08_scheduler.py"),
    ("09", "Paged attention", "test_09_paged.py"),
    ("10", "Prefix caching", "test_10_prefix.py"),
    ("11", "Chunked prefill", "test_11_chunked.py"),
    ("12", "Speculative decoding", "test_12_speculative.py"),
    ("12b", "Structured output", "test_12b_structured.py"),
    ("13", "CUDA graphs", "test_13_graphs.py"),
    ("15", "Profiling", "test_15_profiling.py"),
    ("16", "Triton basics", "test_16_triton.py"),
    ("17", "FlashAttention", "test_17_flash.py"),
    ("18", "Paged attn kernel", "test_18_paged_kernel.py"),
    ("19", "Quantization", "test_19_quantization.py"),
    ("20", "Raw CUDA", "test_20_cuda.py"),
    ("21", "JAX and XLA", "test_21_jax.py"),
    ("22", "Tensor parallelism", "test_22_tp.py"),
    ("23", "MoE / expert parallel", "test_23_moe.py"),
    ("24", "Serving", "test_24_serving.py"),
    ("24b", "Serving agents", "test_24b_agent.py"),
    ("25", "Load testing", "test_25_load.py"),
    ("27", "Routing / disagg", "test_27_routing.py"),
    ("28", "Autoscaling / cost", "test_28_cost.py"),
    ("28b", "Reasoning / TTC", "test_28b_reasoning.py"),
]

GREEN, RED, GREY, DIM, RESET = "\033[32m", "\033[31m", "\033[90m", "\033[2m", "\033[0m"


def run(test_file: str) -> str:
    path = ROOT / "tests" / test_file
    if not path.exists():
        return "unwritten"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(path), "-q", "-m", "not cuda",
         "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=ROOT,
    )
    out = proc.stdout
    if "failed" in out:
        # Unimplemented stubs raise NotImplementedError -- that's homework,
        # not a regression. Anything else failing is a real problem.
        return "todo" if "NotImplementedError" in out else "failing"
    if "NotImplementedError" in out:
        return "todo"

    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
    skipped = int(m.group(1)) if (m := re.search(r"(\d+) skipped", out)) else 0
    deselected = int(m.group(1)) if (m := re.search(r"(\d+) deselected", out)) else 0

    # Everything deselected means the whole lecture is CUDA-only. On a laptop
    # that is expected, not unknown -- say so plainly.
    if deselected and not passed and not skipped:
        return "gpu_only"

    # A lecture is only "done" when its model-backed tests actually RAN.
    # Reporting done off the pure-arithmetic tests alone would tell you a
    # lecture is finished when you haven't written any of it.
    if skipped and not passed:
        return "skipped"
    if skipped:
        return "partial"
    return "done" if passed else "unknown"


def main() -> None:
    print("\nInference Zero-to-Hero -- progress\n")
    counts = {"done": 0, "todo": 0, "failing": 0}

    for num, title, test_file in LECTURES:
        status = run(test_file)
        counts[status] = counts.get(status, 0) + 1
        mark, color, note = {
            "done":     ("[x]", GREEN, ""),
            "partial":  ("[~]", GREY, "arithmetic passes; model tests skipped"),
            "todo":     ("[ ]", GREY, "not implemented yet"),
            "failing":  ("[!]", RED, "FAILING -- something you built broke"),
            "skipped":  ("[-]", GREY, "skipped -- run scripts/fetch_model.py"),
            "unwritten": ("[ ]", DIM, "lecture not written yet"),
            "gpu_only": ("[-]", GREY, "needs a GPU -- rent one (see L09)"),
            "unknown":  ("[?]", GREY, ""),
        }[status]
        line = f"  {color}{mark} L{num:<4}{title:<24}{RESET}"
        print(f"{line}{DIM}{note}{RESET}" if note else line)

    done = counts.get("done", 0)
    print(f"\n  {done}/{len(LECTURES)} lectures verified\n")
    if counts.get("failing"):
        print(f"  {RED}Something you already built is now failing.{RESET}")
        print(f"  {DIM}Regression -- fix before moving on.{RESET}\n")
    elif counts.get("todo"):
        nxt = next((f"L{n} {t}" for n, t, f in LECTURES if run(f) == "todo"), None)
        print(f"  Next up: {nxt}\n")


if __name__ == "__main__":
    main()

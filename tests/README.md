# tests

Correctness gates. "Am I done with this lecture?" has an unambiguous answer.

## Failing tests are your homework

A fresh clone shows failures. **That is the intended state.** Each failure marks
a lecture you haven't built yet:

```
NotImplementedError: M1.2     <- Lecture 06 is waiting for you
```

Tests that fail this way aren't broken — they're the specification. As you
implement each lecture, its tests turn green and stay green.

## Commands

```bash
# What can pass right now, given what you've built
uv run pytest -m "not cuda"

# Just this lecture
uv run pytest tests/test_05_kv_cache.py -v

# Skip anything needing the model download
uv run pytest -m "not cuda and not slow"

# What's left to build, at a glance
uv run pytest -m "not cuda" -q 2>&1 | grep NotImplementedError | sort -u
```

## Markers

| Marker | Meaning |
|---|---|
| `slow` | loads Qwen3-0.6B (~1.2GB on first run) |
| `cuda` | needs an NVIDIA GPU — auto-skipped on a laptop |

## The reference oracle

`conftest.py::reference_greedy` runs HuggingFace's own greedy generation and
pins the result. **Every generation path you write must reproduce it exactly** —
cached, batched, paged, speculative.

Greedy decoding is deterministic, so any divergence is a real bug, never noise.
This is what lets you optimize aggressively later: the moment speed costs you
correctness, a test tells you.

## Progress

Run `uv run python scripts/progress.py` for live status. Statuses mean:

| | |
|---|---|
| `[x]` | verified — implemented and passing |
| `[~]` | arithmetic passes, model tests skipped (fetch the model) |
| `[ ]` | not implemented — this is your homework |
| `[-]` | needs a GPU, or needs the model downloaded |
| `[!]` | **regression** — something you already built broke |

`[!]` is the only one that means something is wrong.

## What passes today

Some tests are green on a fresh clone because they pin *arithmetic* rather
than your implementation — the roofline derivation, Amdahl bounds, cost-per-
million-tokens, sharding math, MoE parameter accounting.

Those aren't filler. They caught three real bugs while this repo was being
written: a wrong percentile assertion, a `code_completion` workload with zero
n-gram repetition (which would have taught the opposite of Lecture 12), and a
block-size table that showed no fragmentation at all.

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

| Lecture | Test | Needs |
|---|---|---|
| 02 arithmetic intensity | `test_02_roofline.py` | ✅ passes today |
| 03 naive generation | `test_03_generation.py` | `model.py::load`, `generate_naive` |
| 04 measuring | `test_04_measuring.py` | ✅ passes today |
| 05 KV cache | `test_05_kv_cache.py` | `generate_cached` |
| 06 sampling | `test_06_sampling.py` | `sampling.py::sample` |
| 07 static batching | `test_07_batching.py` | `generate_batched` |

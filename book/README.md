# Inference Zero-to-Hero

A book about making language models generate tokens fast.

You build an inference engine from scratch — KV cache, continuous batching,
paged attention, prefix caching, speculative decoding — measuring every step,
then read vLLM and understand why it's built the way it is.

**Lectures are short on purpose.** They derive the idea, show the code, and point
outward. The depth lives in the code you run and the numbers you produce.

---

## How to read this

Each lecture gives you four things:

| | |
|---|---|
| **the text** | the idea, derived — 5–10 minutes |
| **a demo** | `uv run python book/code/NN_*.py` — shows the phenomenon |
| **a build** | you implement it in `engine/` |
| **a test** | `pytest tests/test_NN_*.py` — proves it's right |

And a **number that must move**. Record it in `notes/`.

The order matters. Run the demo *before* reading "The idea" — seeing the waste as
a number first is what makes the fix feel inevitable rather than arbitrary.

**Don't skip the predictions.** Several lectures ask you to guess before
measuring. Being wrong in writing is how the intuition forms; skipping ahead to
the answer feels efficient and teaches you much less.

---

## Contents

### Front matter
- [00 — Introduction](00-intro.md) — what this is, why inference is its own discipline, setup

### Part I — Foundations
*Runs on a laptop. No GPU needed.*

- [01 — The two phases](01-the-two-phases.md) — prefill and decode are different problems
- [02 — Arithmetic intensity](02-arithmetic-intensity.md) — the roofline; predict the bottleneck on paper
- [03 — Naive generation](03-naive-generation.md) — write `generate()`, watch it be quadratic
- [04 — Measuring](04-measuring.md) — TTFT, TPOT, percentiles, and why `synchronize()` decides if your numbers are real

### Part II — The engine
*Laptop through L08; rent a GPU from L09.*

- [05 — The KV cache](05-kv-cache.md) — stop recomputing the past
- [06 — Sampling](06-sampling.md) — temperature, top-k, top-p; determinism as a test fixture
- [07 — Static batching](07-static-batching.md) — throughput up, and the padding waste that follows
- [08 — Continuous batching](08-continuous-batching.md) — the scheduler/runner split
- [09 — Paged attention](09-paged-attention.md) — blocks, block tables, fragmentation
- [10 — Prefix caching](10-prefix-caching.md) — content hashing, refcounts, and why context order matters
- [11 — Chunked prefill](11-chunked-prefill.md) — protecting p99 from long prompts
- [12 — Speculative decoding](12-speculative-decoding.md) — draft, verify, accept
- [13 — CUDA graphs](13-cuda-graphs.md) — when launch overhead is the bottleneck
- [14 — Reading vLLM](14-reading-vllm.md) — the capstone: now go read the real thing

### Part III — Kernels
*Needs an NVIDIA GPU.*

- [15 — Profiling](15-profiling.md) · [16 — Triton basics](16-triton-basics.md) · [17 — FlashAttention](17-flash-attention.md)
- [18 — A paged attention kernel](18-paged-attention-kernel.md) · [19 — Quantization](19-quantization.md) · [20 — Raw CUDA](20-raw-cuda.md)

### Part IV — Parallelism
- [21 — JAX and XLA](21-jax-and-xla.md) · [22 — Tensor parallelism](22-tensor-parallelism.md) · [23 — MoE and expert parallelism](23-moe-and-expert-parallelism.md)

### Part V — Production
- [24 — Serving](24-serving.md) · [25 — Load testing](25-load-testing.md) · [26 — Versus vLLM](26-versus-vllm.md)
- [27 — Routing and disaggregation](27-routing-and-disaggregation.md) · [28 — Autoscaling and cost](28-autoscaling-and-cost.md)

### Back matter
- [29 — Contributing](29-contributing.md) — vLLM's test suite, your first PR
- [Appendix A — Glossary](appendix-a-glossary.md)
- [Appendix B — Papers](appendix-b-reading.md)

---

## Field notes

[**field-notes.md**](field-notes.md) collects what practitioners report — real
magnitudes from real deployments, and the places where an optimization
disappointed someone. Books give you the mechanism; these give you the scale.

Use them to sanity-check your own results, and add your own entries as you go.

---

## Companion texts

This book indexes two excellent sources rather than duplicating them:

- **Philip Kiely, *Inference Engineering*** (Baseten, 2026) — the breadth-first
  survey. Lectures cite it by section; read those sections when pointed at them.
- **Aleksa Gordić, [*Inside vLLM*](https://www.aleksagordic.com/blog/vllm)** — a
  top-down read of vLLM V1. **Save it for Lecture 14.** Read before you've built
  a scheduler and it teaches vocabulary; read after and it teaches judgment.

---

## Status

Part I is written. Later parts are in progress — the TOC above is the plan, and
links to unwritten lectures won't resolve yet.

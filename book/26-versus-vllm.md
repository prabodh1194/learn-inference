# 26 — Versus vLLM

**Build:** `bench/compare.py` · **Test:** none — this lecture *is* the test
**Moves:** your ability to explain a performance gap instead of guessing at it
**Prereq:** [25 — Load testing](25-load-testing.md)

---

## The problem

You've built continuous batching, paged attention, prefix caching, chunked
prefill, speculative decoding, and custom kernels.

So: how do you compare to the real thing?

**You will lose.** That is the expected result and it is not the point. The point
is to explain the gap *precisely* — which kernel, which code path, how many
microseconds — using the profiler from Lecture 15.

An engineer who can say "we're 2.3× slower, 60% of which is our attention kernel
and 25% is Python overhead in the scheduler" is far more useful than one who
built something fast by accident.

---

## Run a fair comparison

Most published comparisons are wrong in ways that flatter the author. Control for
all of these:

**Same hardware.** Same GPU, same box, no other tenants. On rented hardware, check
you're not sharing.

**Same model, same precision.** FP16 vs FP16. If vLLM defaults to something else,
pin it explicitly.

**Same workload.** Identical prompts, identical output lengths, identical arrival
pattern. Use your `bench/workloads.py` for both.

**Same measurement point.** Client-observed, through HTTP, both sides. Comparing
your in-process function call against vLLM's HTTP endpoint is not a comparison.

**Warmup, then steady state.** Both engines. Discard the first N requests.

**Same sampling.** Greedy for both, or identical parameters. Speculative decoding
on one side and not the other is a different experiment.

**Report the config.** Both command lines, verbatim, in your notes. A benchmark
whose configuration isn't stated can't be reproduced or trusted.

```bash
vllm serve Qwen/Qwen3-0.6B --dtype float16 --max-model-len 4096 \
  --gpu-memory-utilization 0.9 --disable-log-requests
```

---

## Measure the same three curves

For both engines, on identical workloads:

1. **Single-stream latency** — TTFT and TPOT at batch 1. Tests raw per-step
   efficiency: kernels and launch overhead.
2. **Throughput at saturation** — max sustained tokens/sec. Tests scheduling and
   memory efficiency.
3. **The knee** — from Lecture 25. Tests everything together.

Then run the workloads that isolate specific features: `shared_prefix` for prefix
caching, `long_prefill` for chunked prefill, `code_completion` for speculative
decoding. **A feature-by-feature gap is far more informative than one aggregate
number.**

---

## Explain the gap

Now the actual work. For each gap, form a hypothesis and **verify it with the
profiler** — do not speculate.

Common causes, roughly in order of size:

**Kernels.** vLLM uses FlashAttention/FlashInfer, hand-tuned per architecture.
Your Triton is likely 1.5–3× behind. Profile both; compare kernel times directly.

**Python overhead.** vLLM's hot path is heavily optimized, with much of the
scheduler's work batched or moved off the critical path. Look at CPU time per step
(Lecture 15's first question).

**CUDA graphs.** If you skipped Lecture 13 or capture fewer shapes, you pay launch
overhead they don't.

**Scheduling sophistication.** Better packing, smarter preemption, tuned defaults
that took years of production feedback.

**Memory efficiency.** More blocks available means bigger batches means better
intensity (Lecture 01). Check `gpu_memory_utilization` and your own block count.

For each: **cite a measurement.** "Our attention kernel is 2.1× slower — 340µs vs
162µs per step at 2k context, from `ncu`" is an explanation. "Their kernels are
better" is a guess.

---

## Build it

1. `bench/compare.py` — runs both engines through the same client harness.
2. Produce a table:

| Workload | Yours | vLLM | Ratio |
|---|---|---|---|
| single-stream TTFT | | | |
| single-stream TPOT | | | |
| throughput @ saturation | | | |
| `shared_prefix` TTFT | | | |
| `long_prefill` p99 | | | |
| `code_completion` tok/s | | | |

3. Profile both on the workload with the **largest** gap.
4. Write the gap analysis in `notes/04-production/README.md`: for each row, the
   cause, the evidence, and what you'd do about it.
5. **Pick the biggest one and fix it.** Then re-measure. Closing even one gap with
   a profiler-driven change is worth more than the whole table.

---

## What you should see

**Slower across the board.** Expected.

**Closest on single-stream latency** — fewer moving parts, and your kernels are
reasonable.

**Furthest on saturated throughput** — this is where years of scheduling work
show.

**Possibly competitive on a specific feature.** If your prefix caching is close on
`shared_prefix`, that's a real result. Say so, with the number.

**A gap you can explain line by line.** That's the deliverable.

---

## What this is really teaching

Every performance claim you encounter for the rest of your career — a vendor
benchmark, a blog post, a colleague's "we made it 3× faster" — should now trigger
the same questions:

- Same hardware, model, precision, workload?
- Measured where? Client or in-process?
- Open or closed loop?
- What's the p99, and over how many samples?
- What's the config?

You now know how easy it is to produce a flattering number by accident. That
skepticism is worth more than the benchmark.

---

## Go deeper

- **Kiely §4.5.2** (p.114) — benchmarking tips, including common mistakes.
- **vLLM `benchmarks/`** — read their harness before trusting your own.
- **[Field notes](field-notes.md)** — compiling vLLM from source "doesn't seem to
  increase the performance much," while `--max-num-seqs` mattered a lot.
  Configuration usually beats exotic builds.
- **[SGLang benchmarks](https://github.com/sgl-project/sglang)** — a second point
  of comparison, with different strengths on shared-prefix workloads.

---

## Check yourself

1. You're 2× slower overall but match on `shared_prefix` TTFT. What does that
   tell you?
2. Your gap is much larger at saturation than at batch 1. Which subsystem?
3. Someone claims 3× faster than vLLM. What five questions do you ask?
4. Why is comparing your in-process function to vLLM's HTTP endpoint invalid?
5. From your own table: what is the single highest-value thing to fix, and what
   improvement do you predict?

---

**Next:** [27 — Routing and disaggregation](27-routing-and-disaggregation.md) —
scaling past one replica.

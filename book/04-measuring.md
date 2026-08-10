# 04. Measuring

**Build:** nothing, read `bench/harness.py` · **Test:** `tests/test_04_measuring.py`
**Moves:** your confidence that any of this is real · **Prereq:** [03](03-naive-generation.md)

---

## The problem

You just took your first measurements. Are they real?

That's not rhetorical. There is a specific mistake that makes GPU benchmarks
report speedups that don't exist, it's easy to make, and the resulting numbers
look completely plausible. Most people make it at least once.

This lecture is short. It's here because every number in the remaining 24
lectures depends on getting it right.

---

## The idea

### GPU work is asynchronous

When you call a PyTorch op on GPU, it doesn't run. It gets **queued**, and
control returns to Python immediately. So this:

```python
start = time.perf_counter()
output = model(tokens)          # queued, not executed
elapsed = time.perf_counter() - start   # measures ENQUEUE time
```

...measures how fast Python submitted work. Not how long it took. You'll see
microseconds, conclude something is blazing fast, and be entirely wrong.

The fix is to block until the queue drains:

```python
torch.cuda.synchronize()        # or torch.mps.synchronize()
start = time.perf_counter()
output = model(tokens)
torch.cuda.synchronize()        # <-- wait for it to actually finish
elapsed = time.perf_counter() - start
```

`bench/harness.py::synchronize` handles this for both CUDA and MPS. **Use the
harness rather than hand-rolling `time.perf_counter()`**: this is the single
most common way to fool yourself, and it fools you in the flattering direction.

A related trap: **the first call is always slow.** CUDA context setup, kernel
autotuning, lazy module init. Always run a few warmup iterations and discard
them, or you're benchmarking initialization.

### The four numbers

| Metric | Is | Bound by |
|---|---|---|
| **TTFT** | time to first token | prefill, compute |
| **TPOT** | time per output token, after the first | decode, memory |
| **Throughput** | total tokens/sec across all requests | how well you batch |
| **p50/p90/p99** | latency distribution | your worst users |

TPOT deliberately **excludes** the first token. Prefill and decode are different
phases with different bottlenecks (Lecture 01); averaging them produces a number
that describes neither. `RequestRecord.tpot` measures gaps *between* tokens, so
with 4 tokens there are 3 gaps.

### Percentiles, not the mean

The one non-negotiable reporting rule.

A mean hides its tail completely. 40ms mean with a 2s p99 means one user in a
hundred waits two seconds, and the mean cheerfully reports "40ms." Users
experience the tail, not the average.

The tests demonstrate this directly:

```python
vals = [1.0] * 95 + [100.0] * 5
d = Distribution.from_values(vals)
assert d.p50 == 1.0          # median says "fast"
assert d.p99 > 50.0          # tail says otherwise
assert d.mean < 10.0         # mean hides both
```

**A caveat about your own p99.** Percentiles interpolate between samples, so a
p99 from 100 requests is essentially one data point, noisy and not to be
trusted. Want a real p99? Thousands of requests (Lecture 25). There's a test showing the failure
directly: with 100 samples, a single large outlier lands at p99 *below* 10, the
estimate is biased low, not merely noisy.

### Throughput and latency trade against each other

Bigger batches → better throughput, worse per-user latency. There's no setting
that maximizes both, which is why Lecture 25 plots them as a *curve* and looks
for the knee rather than quoting a single number.

Watch for this whenever someone quotes an impressive tokens/sec: **per user, or
aggregate?** They can differ by 50×, and the ambiguity is doing a lot of work in
most marketing material.

---

## Build it

Nothing new. Instead:

**1. Read `bench/harness.py`.** It's ~360 lines and you'll use it constantly.
Focus on `synchronize()`, `RequestRecord`, and `Distribution`.

**2. Run the tests and read them:**

```bash
uv run pytest tests/test_04_measuring.py -v
```

**3. Prove the async trap to yourself.** Time a forward pass with and without
`synchronize()`. On MPS or CUDA the unsynchronized version will look absurdly
fast. Record both numbers in your notes, seeing the fake number yourself is
worth more than trusting this lecture.

**4. Re-examine your Lecture 03 results.** Did you use the harness? Warm up? If
not, rerun. Better to redo it now than to build five lectures on bad baselines.

---

## Go deeper

- **Kiely §1.4–1.4.2** (p.35–37), TTFT/TPOT and percentiles as product metrics,
  plus end-to-end measurement including network overhead.
- **Kiely §4.5.1–4.5.2** (p.113–114), benchmarking tooling and practical tips.
- **vLLM `benchmarks/benchmark_serving.py`**: the real thing. Note how much of
  it is about *generating realistic load* rather than timing; that ratio is the
  lesson.

---

## Check yourself

1. You time a decode step without `synchronize()` and get 0.1ms. With it, 25ms.
   Which is real, and what did the first number actually measure?
2. Your engine does 2000 tok/s aggregate at batch 64, and 50 tok/s per user.
   Both are true. When does each matter?
3. You measure p99 over 100 requests and get 850ms. Why should you not put that
   in a report?

---

**Part I complete.** You can predict a bottleneck from arithmetic, generate text,
and measure it honestly.

## Next

**[05. The KV cache](05-kv-cache.md)**: the first real optimization, and the
curve you just plotted finally flattens.

```bash
# after implementing generate_cached:
uv run python book/code/naive_bench.py --cached
```

Overlay it on your L03 plot. **Check the slope, not just the speedup**, a
faster number is expected; a *flat line* is the actual result.

# Phase 0 — Baseline and instrumentation

> Status: not started. This file is the template — replace the `_TODO_` markers
> with your real numbers as you go. Keep the predictions even when they're wrong.

Everything downstream is measured against this phase. The goal isn't to build
anything fast; it's to **build the ruler** and prove you can predict a bottleneck
before you measure it.

---

## M0.1 — Load and generate, the slow way

**What I'm trying to make faster:** nothing yet. This is the deliberately naive
version — no KV cache, recomputing the entire forward pass for every single token.

```python
# engine/generate.py :: generate_naive
tokens = tokenizer(prompt, return_tensors="pt").input_ids

for _ in range(max_tokens):
    logits = model(tokens).logits      # <-- the whole sequence, every time
    next_id = logits[:, -1].argmax(-1, keepdim=True)
    tokens = torch.cat([tokens, next_id], dim=-1)
```

The bug-that-isn't: `model(tokens)` re-attends over *all* prior tokens on every
step. Step `n` costs O(n), so generating `N` tokens costs O(N²). Every real engine
exists to avoid this line.

Run it:

```bash
uv run python notes/00-baseline/m01_naive.py
```

**Result:** `_TODO_ tok/s` at 128 tokens.

---

## M0.2 — Build the ruler

`bench/harness.py` came before any engine code, on purpose. The metrics it
reports (book §1.4):

| Metric | Meaning | Bound by |
|---|---|---|
| TTFT | time to first token | prefill — compute-bound |
| TPOT | time per output token | decode — memory-bound |
| p50/p90/p99 | tail latency | what users actually feel |

The one line that separates real numbers from nonsense:

```python
# bench/harness.py :: synchronize
torch.cuda.synchronize()   # or torch.mps.synchronize()
```

Without it you measure how fast you *enqueued* kernels, not how fast they ran.
GPU work is asynchronous; `time.perf_counter()` around an unsynchronized call
will happily report a 50x speedup that doesn't exist.

**Why percentiles, not the mean:** a mean hides the tail completely. A service
with a 40ms mean and a 2s p99 is a service where 1 in 100 users waits two seconds.
Report `p99` in every result or you're measuring the wrong thing.

---

## M0.3 — Plot the pain

**Prediction (write this BEFORE running):**
per-token time should climb roughly linearly with position, because step `n`
attends over `n` prior tokens.

_TODO_: was it linear? Where did it deviate, and why? (Hint: at short lengths,
fixed overheads dominate and the curve looks flat. That's not a contradiction —
it's the regime where the quadratic term hasn't caught up yet.)

```bash
uv run python notes/00-baseline/m03_curve.py   # -> bench/results/m03-time-per-token.png
```

| seq len | naive tok/s | time/token at end |
|---|---|---|
| 128 | _TODO_ | _TODO_ |
| 256 | _TODO_ | _TODO_ |
| 512 | _TODO_ | _TODO_ |
| 1024 | _TODO_ | _TODO_ |

**This curve is the whole motivation for M1.1.** Keep the plot; you'll overlay
the cached version on top of it and the contrast is the payoff.

---

## M0.4 — Roofline, by hand

Reproduce the book's arithmetic-intensity calculation (§2.4, Figs 2.14–2.18) with
Qwen3-0.6B's real dimensions.

**The model (book Fig 2.16–2.18)**, for one attention step with sequence length
`N` and head dim `d`, unoptimized:

```
memory movement = 8N² + 8Nd   bytes      (FP16: 2 bytes/value)
compute         = 4N²d + 3N²  ops
intensity       = compute / memory
```

The book's worked example (N=4096, d=128) gives **62 ops:byte** against an H100's
**295 ops:byte** ratio (989 TFLOPS ÷ 3.35 TB/s) — nearly 5x below the ridge, hence
memory-bound.

**Fill in for your hardware:**

| Device | peak FLOP/s | bandwidth | ops:byte ridge |
|---|---|---|---|
| M1 (8GB) | _TODO_ | _TODO_ | _TODO_ |
| rented _TODO_ | _TODO_ | _TODO_ | _TODO_ |

**Prediction:** prefill lands _TODO_ the ridge (compute-bound); decode lands
_TODO_ the ridge (memory-bound).

Why they differ, in one sentence each:
- **Prefill** loads the weights once and does large matrix-matrix multiplies over
  the whole prompt → lots of compute per byte moved → high intensity.
- **Decode** re-loads the entire weight matrix to generate *one* token via a
  matrix-vector product → almost no compute per byte → low intensity.

```bash
uv run python notes/00-baseline/m04_roofline.py   # -> bench/results/m04-roofline.png
```

---

## ✅ Gate

You can state, **with your own measured numbers**, why decode is memory-bound and
prefill is compute-bound.

If you can't say it without looking it up, don't move on — this single distinction
explains nearly every optimization in Phases 1 and 2. Continuous batching works
because it adds compute to a memory-bound phase for free. Quantization works
because it moves fewer bytes. Speculative decoding works because it turns
sequential memory-bound steps into one parallel compute-bound step.

---

## What surprised me

_TODO_ — the most valuable section. Wrong predictions belong here, unedited.

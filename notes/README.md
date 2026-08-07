# Notes

The learning record. Karpathy's format: **prose and runnable code interleaved**,
where the code *is* the explanation.

## The pattern

One directory per phase. Each milestone gets a `.md` note and, where it helps, a
matching runnable `.py` you can execute top-to-bottom:

```
notes/
  00-baseline/
    README.md          the phase writeup, in order
    m01_naive.py       runnable: naive generate, no cache
    m03_curve.py       runnable: produces the quadratic-pain plot
    m04_roofline.py    runnable: arithmetic intensity by hand
```

Every code block in a note should be **copy-pasteable and actually run**. If a
snippet has drifted from the code, the note is wrong — fix it.

## Note structure

Each milestone note answers five questions, in this order:

1. **What am I trying to make faster?** State the bottleneck *before* you touch it.
2. **What did I measure first?** The baseline number. No baseline, no claim.
3. **What did I build?** The code, with the non-obvious lines explained.
4. **What moved?** Before/after, from `bench/results/*.json`. Include p99, not just the mean.
5. **What surprised me?** The most valuable section. Wrong predictions go here — keep them, don't edit them out.

## Writing rules

- **Predict before you measure.** Write the guess down first. Being wrong in
  writing is how the intuition actually forms.
- **Keep the failures.** The kernel that was slower, the optimization that did
  nothing on the wrong workload. Deleting these deletes the lesson.
- **Numbers or it didn't happen.** Every claim cites a result JSON.
- **Explain to someone one step behind you.** That framing is why this doubles
  as teaching material.

## If these become a video series

They're already the storyboard — the format maps cleanly onto episodes:

| Note section | On screen |
|---|---|
| the bottleneck | motivate: why is this slow? |
| baseline measurement | run it, watch it be slow |
| the build | live-code the fix |
| what moved | rerun the benchmark, show the plot |
| what surprised me | the honest bit people remember |

The thing that makes a series credible isn't knowing everything up front — it's
**deriving it on screen with real measurements**. That's what these notes capture,
and it's why writing them as you go matters more than writing them well.

## Index

| Phase | Note | Milestones |
|---|---|---|
| 0 | [00-baseline](00-baseline/) | M0.1–M0.4 — naive generate, bench harness, roofline |
| 1 | [01-engine](01-engine/) | M1.1–M1.9 — KV cache → paged attention → specdec |
| 2 | [02-kernels](02-kernels/) | M2.1–M2.6 — profiling, Triton, flash/paged attention, CUDA |
| 3 | [03-parallelism](03-parallelism/) | M3.1–M3.5 — JAX, sharding, TP, MoE |
| 4 | [04-production](04-production/) | M4.1–M4.6 — serving, load testing, vs. vLLM, P/D |

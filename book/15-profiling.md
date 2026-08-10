# 15. Profiling

**Build:** `kernels/profile_engine.py` · **Test:** `tests/test_15_profiling.py` (cuda)
**Moves:** nothing; this decides what's *worth* moving · **Prereq:** [14. Reading vLLM](14-reading-vllm.md)

> **NVIDIA GPU required** for Nsight. `torch.profiler` works on MPS with reduced
> detail, so you can practise the method on a laptop.

---

## The problem

Part III is about making kernels faster. Before writing a single one, you need to
know **which** kernel, because the honest answer is usually not the one you'd
guess.

Optimizing without profiling is how people spend a weekend on a kernel that
accounts for 3% of runtime and report a "12% improvement" that's 0.4%.

Amdahl's law, stated as a rule: **you cannot make something faster than the time
you didn't spend in it.**

---

## The idea

Three questions, in order. Each has a different tool.

### 1. Am I GPU-bound or CPU-bound?

The one Lecture 13 already raised. If the GPU is idle waiting for Python, no
kernel work helps.

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        decode_step()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
```

Compare `Self CPU total` against `Self CUDA total`. CPU much higher means you're
launch-bound, go back to Lecture 13.

### 2. Where does GPU time actually go?

Same table, sorted by CUDA time. Expect roughly:

| Operation | Share of decode | Why |
|---|---|---|
| Linear layers (QKV, O, MLP) | 60–80% | most of the weights |
| Attention | 10–25% | grows with context |
| Norms, RoPE, elementwise | 5–15% | many small kernels |
| Sampling | <5% | one small op |

**The exact split is what matters, and it's model- and context-dependent.** A long
context shifts weight toward attention; a short one toward the MLP. Optimizing
attention when your workload runs 512-token contexts is a common misallocation.

### 3. Is *this* kernel efficient?

Once you've picked a target, Nsight Compute tells you whether it's near a
hardware limit:

```bash
ncu --set full -o profile python bench_decode.py
nsys profile -o timeline python bench_decode.py
```

- **Nsight Systems (`nsys`)**: timeline. Gaps, overlap, CPU/GPU interaction.
  Answers "what is the machine doing?"
- **Nsight Compute (`ncu`)**: per-kernel. Occupancy, memory throughput, achieved
  vs. peak. Answers "is this kernel good?"

The number to look for is **achieved memory bandwidth as a fraction of peak**.
Lecture 02 said decode is memory-bound; a well-written decode kernel should be at
70–90% of peak bandwidth. At 30%, there's real headroom. At 85%, you are close to
the roofline and should look elsewhere.

That's the roofline from Lecture 02, now measured instead of predicted.

---

## The method

The discipline that makes profiling useful rather than a hobby:

1. **Measure first.** Get a baseline with `bench/`.
2. **Profile.** Rank kernels by total time.
3. **Pick the top item.** Not the interesting one: the top one.
4. **Check the ceiling.** If it's at 85% of peak bandwidth, the win is small;
   move on. This step prevents most wasted effort.
5. **Optimize.** One change.
6. **Re-measure end to end.** Kernel-level wins that don't show up in
   `bench/` results aren't wins.
7. **Re-profile.** The bottleneck moved. Start again.

Step 6 is the one people skip. A kernel 2× faster that was 4% of runtime buys you
2%, which is inside your measurement noise. **Only end-to-end numbers count.**

### Warmups and steady state

Same trap as Lecture 04, one level down. First iterations include CUDA context
setup, autotuning, and lazy initialization. Always discard warmup iterations, and
profile a **steady-state** decode loop rather than a cold start: that's what your
server actually spends its life doing.

---

## Build it

1. Write `kernels/profile_engine.py`: run your engine under `torch.profiler`,
   emit a ranked kernel table.
2. Profile **decode at batch 1** and **decode at batch 32**. The rankings will
   differ, note how.
3. Profile **prefill** separately. Different phase, different bottleneck,
   different ranking (Lecture 01).
4. Run `ncu` on your top kernel. Record achieved bandwidth vs. peak.
5. **Write down your top five kernels with their time share** in
   `notes/02-kernels/README.md`.

That list is your work queue for Lectures 16–20. Every optimization from here
must cite it.

**Predict before you profile:** which kernel is #1 at batch 1? At batch 32? Write
both down.

---

## What you should see

**At batch 1**, launch overhead may dominate everything: the Lecture 13 signature.

**At batch 32**, real compute takes over and linear layers lead.

**In prefill**, attention matters much more than in decode, and its share grows
quadratically with sequence length.

**Achieved bandwidth well below peak** in the naive PyTorch attention you wrote in
Part II. That gap is Lecture 17's target.

---

## Go deeper

- **[NVIDIA Nsight Compute docs](https://docs.nvidia.com/nsight-compute/)**: start
  with the "Speed of Light" section; it's the roofline in tool form.
- **[PyTorch Profiler recipe](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)**:
  including the Chrome trace export, which is easier to read than the table.
- **Kiely §4.5.3** (p.114), profiling in an inference context.
- **Kiely §4.1.1–4.1.2** (p.98), CUDA kernels and kernel selection, useful
  vocabulary before Lecture 16.

---

## Check yourself

1. Your top kernel is 8% of runtime. You make it 3× faster. End-to-end gain?
2. A kernel hits 88% of peak memory bandwidth. Is optimizing it worthwhile?
3. Why do decode and prefill need separate profiles?
4. Batch 1 and batch 32 give different rankings. What changed?
5. From your table: what will you optimize first in Lecture 16, and what
   end-to-end improvement do you predict?

Write that prediction down. Lecture 20 checks it.

---

## Next

**[16. Triton basics](16-triton-basics.md)**: write your first kernel.

Before you start, use your L15 table to **predict the end-to-end gain**:

```bash
uv run python -c "from kernels.profile_engine import amdahl_speedup; \
  print(amdahl_speedup(share=0.04, speedup=2.0))"
```

A 4% kernel made 2× faster buys 2.0%. Small is the *correct* answer here, L17
and L18 are where the big numbers live.

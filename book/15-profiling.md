# 15. Profiling

**Build:** `kernels/profile_engine.py` · **Test:** `tests/test_15_profiling.py` (cuda)
**Moves:** nothing; this decides what's *worth* moving · **Prereq:** [14. Reading vLLM](14-reading-vllm.md)

> **NVIDIA GPU required** for Nsight. `torch.profiler` works on MPS with reduced
> detail, so you can practise the method on a laptop.

---

## The problem

Part III is about making kernels faster. A **kernel** is a small program the
chip runs: one step of the model's math, one matmul, one normalization, one
add. The forward pass is dozens of them, and they run one after another, so
the total time is the sum of their individual times. Before writing a single
one, you need to know **which** kernel, because the honest answer is usually
not the one you'd guess.

Optimizing without profiling is how people spend a weekend on a kernel that
accounts for 3% of runtime and report a "12% improvement" that's 0.4%. Here is
that arithmetic, because it is the whole lesson. The 12% applies only to the
kernel's own slice of the step, not to the step:

```
0.03  ×  0.12  =  0.0036  =  0.36%  ≈  0.4%    of the step overall
```

Twelve percent of the kernel is a third of a percent of the step. Nobody sees
the step get faster, and a weekend is gone.

Amdahl's law, stated as a rule: **you cannot make something faster than the
time you didn't spend in it.** A kernel can only speed up its own slice; the
other 97% of the step runs at its old speed, untouched, forever.

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
- **Nsight Compute (`ncu`)**: per-kernel. Occupancy (how many warps, the chip's
  fixed 32-thread groups, it can keep resident and ready at once), memory
  throughput, achieved vs. peak. Answers "is this kernel good?"

The number to look for is **achieved memory bandwidth as a fraction of peak**.
Before you can read it, you need to know what it's made of: the profiler
doesn't *measure* it, it computes it from two other numbers.

#### The metric is a division, not a reading

```
fraction   =   achieved bandwidth   ÷   peak bandwidth

achieved   =   bytes the kernel moved   ÷   seconds it ran
               (DRAM traffic counter)        (kernel duration)

peak       =   the chip's theoretical maximum — what the DRAM bus can
               physically deliver:  bus width × transfer rate
```

The fraction is what `ncu` prints as a percentage. NVIDIA's docs call it
`pct_of_peak_sustained_elapsed` — "how close a portion of the GPU reached to
peak rate": every counter has a peak rate in the profiler's chip database, and
the percentage is the counter divided by it. Your job is to read the
percentage, know what the division means, and decide whether the answer is
"done" or "loafing".

#### Worked example: your decode GEMM on a 3090

Decode reads all the weights per token (Lecture 02): 840 MiB = 0.881 GB. The
profiler clocks the kernel; the division does the rest:

| Kernel duration | Achieved (0.881 GB ÷ time) | Fraction of 936 GB/s peak | Verdict |
|---|---|---|---|
| 3.0 ms | 294 GB/s | 31% | **loafing — 3× headroom** |
| 2.0 ms | 441 GB/s | 47% | mid |
| 1.1 ms | 801 GB/s | 86% | **near the roofline — stop** |
| 1.0 ms | 881 GB/s | 94% | physically close to done |

Same kernel, same weights, same GPU. The only thing that changed is the
duration, which is the only thing you control. That's the whole point of the
fraction: it converts "how long did it take" into "how close to the physical
ceiling am I". At 31% the memory system could deliver 3× more bytes — the
kernel is leaving time on the table. At 86% you're at the roofline; optimizing
further buys almost nothing.

#### Same kernel, your machine

The M1's unified memory peaks at ~68 GB/s — 14× less than the 3090. Same
0.881 GB of weights:

```
best possible:  0.881 GB ÷ 68 GB/s = 13 ms per token   (100%, unreachable in practice)
```

| Kernel duration | Achieved | Fraction of 68 GB/s peak |
|---|---|---|
| 18 ms | 49 GB/s | 72% — fine |
| 26 ms | 34 GB/s | 50% — headroom |
| 43 ms | 20 GB/s | 30% — loafing |

The fraction is why "my Mac does 25 tok/s" and "a 3090 does 100 tok/s" can be
the same kernel at the same distance from their own ceilings — and why you
can't see the headroom until you divide.

#### What the fraction looks like in real output

Here is an actual `ncu` section for a memory-bound kernel, from the University
of Wisconsin's profiling guide:

```
Memory Throughput                 %        43.35
DRAM Throughput                   %        32.94
Compute (SM) Throughput           %         4.42
```

Read it in one line: the kernel is memory-bound (43% memory vs 4.4% compute)
but only at a third of peak DRAM — so there's headroom, and the cause is
usually not the DRAM at all but *latency*: the GPU can't keep enough
transactions in flight. `ncu`'s speed-of-light section says exactly that:

> Achieved compute throughput and/or memory bandwidth below 60.0% of peak
> typically indicate latency issues.

That's where the book's thresholds come from: 70–90% means the memory system
is genuinely saturated; below ~60% you're usually latency-bound, and the fix
is occupancy and parallelism, not "faster math".

#### The same GPU, two access patterns — proof it's the kernel

The cleanest demonstration that the fraction is a kernel property, not a
hardware spec: measured on the same Jetson GPU, with the peak fixed, the
access pattern alone moves the achieved number:

| Access pattern | Achieved | Fraction of peak |
|---|---|---|
| sequential read | ~65 GB/s | near peak |
| random (bitonic) | 3–12 GB/s | ~5–18% |

Same chip, same working set, same peak. Random access collapses achieved
bandwidth 5–25× — DRAM latency, exposed. "Achieved bandwidth" is *earned*,
not given: the peak is physics, the achieved is your kernel.

#### One trap

DRAM traffic counts bytes that left the chip; cache hits don't count. A
flash-attention kernel can sit at 20% of peak and be perfect — its data never
left the SRAM. So the fraction only means "close to done" for kernels that
*should* be DRAM-bound: exactly decode GEMMs, which re-read all the weights
every token. That's why the book leans on it here — and why it's only one of
three checks (with occupancy and the roofline) for anything else.

That's the roofline from Lecture 02, now measured instead of predicted.

??? question "What does 'achieved bandwidth as a fraction of peak' even mean?"
    It's a division, not a reading: bytes moved ÷ kernel duration = achieved;
    achieved ÷ the chip's spec peak = the percentage `ncu` prints. It
    converts "how long did it take" into "how close to the physical ceiling
    am I" — 30% = headroom, 85% = done. Trap: cache hits don't count as DRAM
    traffic, so a low fraction can be fine for kernels that shouldn't be
    DRAM-bound. Worked numbers above; full answer in
    [qa.md](qa.md#what-does-achieved-memory-bandwidth-as-a-fraction-of-peak-mean)

---

## The method

The discipline that makes profiling useful rather than a hobby. One thing to
state up front, because it's easy to lose in the bandwidth talk above:

> **The objective is less end-to-end decode time — never "higher memory
> utilization" for its own sake.** Utilization is the *instrument*, not the
> goal: it tells you whether a kernel can still get faster. For a
> memory-bound kernel the two coincide — at 85% of peak you *are* as fast as
> physics allows — which is why the ceiling check below works. But the score
> is always the `bench/` number, and step 6 exists to say so.

??? question "Is the objective to improve memory utilization?"
    No — the objective is less end-to-end decode time. Utilization is the
    instrument: it tells you whether a kernel can still get faster. The
    confusion is natural because for a memory-bound kernel the two coincide
    (85% of peak = as fast as physics allows), which is why step 4 uses it as
    the "stop" check. And at 30%, the fix is usually *not* "make memory
    busier" — it's latency-hiding: more occupancy, more parallelism.
    [Full answer](qa.md#is-the-objective-to-improve-memory-utilization)

1. **Measure first.** Get a baseline with `bench/`.
2. **Profile.** Rank kernels by total time.
3. **Pick the top item.** Not the interesting one: the top one.
4. **Check the ceiling.** If it's at 85% of peak bandwidth, the win is small;
   move on. This step prevents most wasted effort.
5. **Optimize.** One change.
6. **Re-measure end to end.** Kernel-level wins that don't show up in
   `bench/` results aren't wins.
7. **Re-profile.** The bottleneck moved. Start again.

Step 6 is the one people skip. A kernel 2× faster that was 4% of runtime buys
you 2%, which is inside your measurement noise. Every term in that claim:

```
old step   =  96% (everything else)   +  4% (the kernel)        =  100%
new step   =  96%                     +  4% / 2  (kernel halved) =  98%
speedup    =  100 / 98                =  1.0204                  ≈  +2%
```

**Only end-to-end numbers count.**

### Warmups and steady state

Same trap as Lecture 04, one level down. First iterations include CUDA context
setup (the GPU-side bookkeeping that only comes into being on first use),
autotuning, and lazy initialization. Always discard warmup iterations, and
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

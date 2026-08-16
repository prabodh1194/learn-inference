# 14f. CPU/GPU overlap is engineered

> Your engine's decode loop spends its life like this: CPU prepares work,
> GPU executes it, CPU waits, repeat. The h3.c loop has the same shape, twenty
> denoise steps deep — and its author treats the CPU/GPU boundary as a
> **scheduling problem with a budget**, not a place where you hope things
> overlap. Two mechanisms do the work: splitting the command buffer (the queue of GPU commands the CPU encodes), and
> moving bookkeeping off the GPU entirely.

## The split command buffer

The objective here is to keep the CPU building the next chunk while the GPU
runs the current one — so the two never serialize.

The per-step work is one long ordered chain: condition rows → AdaLN schedule →
50 blocks (each: QKV, GQA, output projection, MLP, residual) → heads. If the
CPU submits the whole chain and then waits, the GPU sits idle while the CPU
builds the *next* step's chain — and the CPU and GPU serialize:

```
naive:   [CPU builds step n] [GPU runs step n] [CPU builds step n+1] [GPU runs...]
```

The fix is to **split the command buffer at a fixed depth and let the CPU
start building the next chunk before the GPU finishes the current one**
(`h3_gpu.c`, `h3_gpu_continue`). The Metal semantics make this cheap: an
MPSCommandBuffer (Metal's command-buffer class) is a queue; the CPU can commit chunk *n* (without waiting)
and immediately begin encoding (building the commands for) chunk *n+1* into the same buffer, while the GPU
runs chunk *n*. The wait happens only at the end of the loop.

The measured gain is the honest smallness you should expect from a
well-engineered loop: **0.5–1.8% end-to-end** on the M5 Max at the default
60% split depth (the split point after 60% of the blocks — the tuned sweet
spot between CPU head-start and commit cost), with the M3 Max gaining ~1.2% only at 30/50 depth and
**regressing** at 24/40. Quoted: *"the M3 Max gained 1.2% only with the 30/50
split, while 24/40 lost time."* The reason is mechanical: splitting costs two
things — an extra commit boundary per step, and the loss of the GPU's freedom
to reorder work within the chunk. On a machine whose CPU is fast relative to
its GPU, the split buys little; on the M5, the CPU work per step (encoding,
the sampler, the audio bus) is heavy enough that overlap pays.

The lesson, stated sharply: **the async structure of Metal (and CUDA, and
CUDA graphs — a captured set of kernels replayed as one unit) gives you the *option* of overlap — it does not give you the
overlap.** If the CPU can't get ahead of the GPU, the boundary does nothing;
and every split you add has a cost. This is Lecture 13's capture-vs-replay
tradeoff at the buffer level: commit-without-wait is the graph, and the split
depth is the capture granularity you tune.

## The GPU sampler, retired

The objective here is to remove a per-step synchronization point by moving
the sampler off the GPU.

The first version of the sampling code ran the sampler on the GPU and
**read the sampled tokens back to the CPU** — a synchronization point per
step. The README's diagnosis is the kind every profiler should produce:
the readbacks cost 0.1–0.3% of runtime, *and* they pinned (locked in place so the GPU can read it) ~136 MB of host
state to the GPU's memory every step. The fix was structural: sampling moved
to the CPU entirely. The GPU writes the logits; the CPU samples. The wait is
gone because there is nothing to wait for.

??? question "Why is a GPU sampler with a readback slower than a CPU sampler?"
    A GPU kernel that writes one value and then gets *read back* is the worst
    shape in GPU programming: the kernel itself is trivial, the readback forces
    a full pipeline drain (Lecture 04's synchronize, enforced), and every step
    pays both. A CPU sampler reads the logits tensor once — the same bytes the
    readback would have fetched — and does the rest of its work while the GPU
    is already busy with the next step. Moving it off the GPU turns a
    *synchronization point* into *ordinary traffic*.
    [Full answer](qa.md#why-is-a-gpu-sampler-with-a-readback-slower-than-a-cpu-sampler)

The deeper move is the one the numbers hide: sampling on the CPU means the
**GPU never stalls to hand over control**, and the CPU's sampling time hides
under the next step's GPU work instead of standing in its way. The 136 MB is
the clue — the readback wasn't just a stall, it was *residency*: host memory
pinned for the GPU's benefit, held every step for a value read once.

## The rule behind both: synchronize before you time

None of this is measurable without the book's timing rule (Lecture 04, and
the `bench/harness.py` discipline): GPU work is async; a `time.perf_counter`
around a submit measures *enqueue*, not execution. The 0.5–1.8% split gain
and the 0.1–0.3% sampler cost are only real if every timing includes the
`commitAndContinue`-style drain — which is why h3.c's timing harness runs the
full pipeline and times generation end-to-end rather than per-kernel. The
split's gain is *specifically* an end-to-end gain; per-kernel timing would
show it as nothing at all, because per-kernel timing is where it doesn't
exist.

## Check yourself

1. The 24/40 split regressed on the M3 Max. What two costs does any split
   add, and which one explains why a *shallow* split hurts the M3?
2. The GPU sampler cost 0.1–0.3% and 136 MB of pinned host state. Which of
   those is the real reason it was removed, and why?
3. Why would per-kernel timing show the split command buffer as "no gain"?
   What measurement would you run instead?

## Next

**[14g. Quantization with receipts](14g-quantization.md)**: int8 lands on the
DiT — and every claim comes with a number attached.
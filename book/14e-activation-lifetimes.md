# 14e. Activation lifetimes as a memory budget

> Lecture 14c was about weights — the budget you can plan. Activations are the
> budget that surprises you: they live per-step, scale with the sequence, and
> pile up invisibly until peak memory is whatever the worst moment was. The
> DiT's (diffusion transformer's) trick is to treat activation memory as a **lifetime-scheduling
> problem**: read when each tensor is born and when it dies, then overlap the
> dead ones. Three moves in h3.c, each worth real MiB at 20 steps per
> generation.

## The budget, derived

At 512-class (the 512×512 video geometry), the denoise grid is about **2,835 rows × 5376 hidden**, BF16 (brain float16: 8-bit exponent, 7-bit mantissa):

```
per buffer   =  2835 × 5376 × 2 bytes
             =   30,481,920 B
             ≈       29.07 MiB
```

Every per-block activation that h3.c eliminates is a multiple of ~29 MiB per
buffer — and the DiT keeps several alive per block: the QKV input (the query/key/value projection's output), the
attention output, the MLP intermediate, the AdaLN (adaptive layer norm — the per-row shift/scale schedule from 14d) schedule, the skip. The
naive budget is "however many buffers the block holds, at the worst block, at
the worst moment." The optimized budget is "however many buffers are *alive*
at that moment" — and aliveness is the tool.

## Move 1: the alias assignment

The objective here is to show three named buffers sharing one allocation —
legal because their lifetimes never overlap.

The three lines of code that saved the most MiB for the least complexity
(`h3_gpu.c:1360-1362`):

```c
gpu->qkv = threadgroup_alloc(threadgroup_buf_size * 2);
gpu->attention_heads = gpu->qkv;
gpu->mod_mlp = gpu->qkv;
```

Three named buffers, two allocations. The trick is in the lifetimes:

```
 qkv           born: input projection        dies: after GQA reshape
 attention_heads born: after GQA reshape      dies: after output projection
 mod_mlp        born: after attention output  dies: end of block
```

At 512-class, the QKV buffer holds `qkv → attention_heads → mod_mlp` in
sequence — each successor born only after its predecessor died, so they share
one allocation. The measured savings, quoted: **61.25 MiB** at 512-class,
**99.63 MiB** at 864-class (the 864×864 video geometry), from this single
alias — three lines of code buying the engine's largest activation saving —
versus the naive layout that would allocate all three.

Why is this legal? Because the alias is *time-disjoint*, not just
size-compatible: the code that writes `attention_heads` is exactly the GQA
reshape (grouped-query attention's data shuffle — the repacking of the QKV buffer into per-head views), which completes before the output projection reads it, which
completes before `mod_mlp` is written. The safety comes from reading the
kernel as a dataflow graph and checking the interleaving. The risk — and why
it's an *engineering* artifact, not a code smell — is that the alias is only
valid as long as the fused schedule stays exactly this interleaved: it is one
of the first things that would break if a future block reorders the GQA
against the MLP.

## Move 2: the token-reduction bypass in the QKV tail

The objective here is to reuse a buffer region that is dead after pooling as
free scratch — saving an allocation at zero cost.

The token-reduction path (14h) reduces the attention grid by pooling rows
2×2. The pooled input rows are produced by a row-pooling kernel — and that
kernel's output is written **into the tail of the QKV buffer**, the region
that attention no longer reads once the sequence is pooled:

```
full QKV:  [ rows 0..2834  |  tail: pooled input rows  ]
                       ↑ dead after pooling
attention reads the pooled rows from the tail
```

This is the same lifetime discipline, one level weirder: **dead scratch is
storage**, and the dead tail of a buffer is free memory that happens to be
contiguous. It saves whatever the pooled rows would have cost as a separate
allocation, at zero allocation cost, because the tail was dead anyway.

## Move 3: the per-block AdaLN schedule

The objective here is to bound peak activation memory by one block's working
set instead of the number of blocks.

The AdaLN schedule (14d) computes ~498 MiB of shift/scale per block. It is
allocated per block and **released before the next block starts** — which
sounds like ordinary scoping until you notice what it implies: peak activation
memory is bounded by *one* block's working set, not by the number of blocks,
and the resident peak never accumulates across the 50-block loop. The naive
mistake would be to precompute all 50 schedules upfront (they're only
dependent on condition + step, not on the block's own output) — 50 × 498 MiB
≈ 24.3 GiB of resident memory for a value each block regenerates anyway.

The discipline, generalized: **allocate by lifetime, not by name.** The
questions that turn a memory audit into a schedule are:

1. When is this tensor first written?
2. When is it last read?
3. What else is alive in between?

Anything whose answer to (2) precedes another tensor's answer to (1) can
share. This is the book's Lecture 09 budget lesson with the sequence replaced
by the block loop: the peak is the *intersection* of lifetimes, not the union
of allocations.

## Check yourself

1. The alias `qkv = attention_heads` is only valid if the GQA reshape fully
   completes before the output projection reads `attention_heads`. What
   instruction in the DiT's schedule makes that true, and what would you check
   first if it ever started failing?
2. Why can the token-reduction path write its pooled input into the *tail* of
   the QKV buffer instead of allocating a new one?
3. The AdaLN schedule depends only on (condition, step), never on the block's
   own activations. Why does that make precomputing all 50 schedules the
   tempting mistake — and what does it cost?

## Next

**[14f. CPU/GPU overlap](14f-cpu-gpu-overlap.md)**: the encoding work hides
under the GPU's grind — and the trick is a split command buffer, not luck.
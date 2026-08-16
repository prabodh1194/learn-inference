# 14d. Fusion: launch and memory trades

> Lecture 13 showed that CPU launch overhead can dominate a decode step with
> thousands of tiny kernels. The DiT in h3.c faces the same enemy, twenty
> steps at a time, and answers by **fusing** — merging kernels so a value
> never leaves the chip and a dispatch (one kernel launch on the GPU's command queue) never happens. The common instinct is
> that fusion is a *speed* trick. Measured closely, the bigger wins here are
> **memory and launch overhead** — and the discipline that keeps fusion safe
> is a rounding boundary, not a speedometer.

## What fusing actually buys

Fusing kernel B into kernel A can save any of three things:

1. **A dispatch** — one less kernel launch on the command queue (Lecture 13).
2. **A global read+write** — B's input would have round-tripped through
   global memory; now it stays in registers or threadgroup memory (on-chip SRAM shared within one workgroup — fast, but only a few tens of KiB).
3. **A separate pass** — if A and B have the same grid, one launch can do both
   bodies, and the grid is scheduled once.

A 5376-wide hidden state, BF16 (brain float16: 8-bit exponent, 7-bit mantissa), is ~10.5 KiB per row of activations (5376 × 2 bytes = 10,752 B = 10.5 KiB). Every
fusion that keeps one intermediate on-chip saves writing ~10.5 KiB to global
memory and reading it back — per row, per block, per denoise step, times 20
steps. The savings compound exactly like the traffic model of Lecture 02: a
value that never lands in DRAM costs zero DRAM time.

## Case 1: the gate fused into the AdaLN

The objective here is to remove the gate's separate kernel pass by computing
the gate inside the AdaLN kernel.

The DiT's residual block computes an AdaLN (adaptive layer norm) schedule:
per-row shift/scale derived from the condition and the step, applied before
the attention and MLP sublayers. There is also a *gate*: a per-row factor that
scales the block's contribution, and a skip connection carrying the block's
input. In the naive layout the gate is a tiny separate kernel that reads the
skip from global memory.

The fusion (`h3_shaders.metal`, the `Gate+AdaLN` kernel): the gate is computed
inside the AdaLN kernel. What that eliminates, item by item:

- **One dispatch** — the gate kernel disappears.
- **One global reread** — the skip was read twice (once by the gate kernel,
  once by the AdaLN); now once. The skip is the residual stream: width 5376,
  every row, every block.
- **One grid schedule** — the two had the same row-grid and the same thread
  count, so they merge cleanly.

The net is small per dispatch and large per loop: the README reports ~99
dispatches eliminated per forward pass for the int8 fused path — about 99 × 20 ≈ 2,000 dispatches not launched per generation — and the fused kernel is the default because the numbers are simply better on both M5 and M3.
This is Lecture 13's launch-overhead lesson, applied by hand: **count the
dispatches before you count the FLOPs.**

## Case 2: the cross-block AdaLN carry

One fusion is not a merge but a *carry*: the AdaLN schedule of the next block
is computed inside the current block's kernel and handed over in threadgroup
memory, so the next block starts with its schedule already resident
(`h3_shaders.metal`, "AdaLN cross-block carry"). The catch is structural —
the carry creates a **loop-carried dependency** across kernel invocations:

```
block n's kernel writes next_shift/next_scale
           ↓ (threadgroup memory, same dispatch)
block n+1's kernel reads them
```

That only works when the two blocks run back-to-back in the same dispatch
chain with the same grid. If a control-flow boundary splits them (the
schedule-thinning of 14h, the streamed forward of 14c, a debug flag), the
carry is silently invalid. h3.c's fix is the discipline of 14h: **every
optimization carries an env-var escape hatch**, and this one has an explicit
restart — the carry is recomputed at any block that is not a direct successor.
Fusion that *changes the control-flow contract* is the fusion most likely to
bite later.

## Case 3: the fused final heads

The objective here is to run the three final heads in one dispatch — and
convert their shared intermediate to BF16 once instead of three times.

The last block's output goes through three heads (video, audio, and the video
VAE's conditioning embedding), each a linear layer. The naive layout: one
kernel computes all three logits into three BF16 tensors in global memory;
three separate kernels consume them. The fused version runs the three heads in
one dispatch and converts the intermediate to BF16 **once, at load**:

```
naive:  3 linear heads → 3 global tensors → 3 consumers
fused:  1 dispatch, 3 heads → 1 global tensor (BF16, converted once)
```

The measured savings are quoted per geometry: 18.8 MiB saved at 512-class for
the video head alone, and the combined heads fusion saves 37.5 MiB (512-class)
and 58.9 MiB (864-class) of intermediate traffic — bytes that would otherwise
round-trip global memory, saved at every one of the 20 denoise steps. Why does the *conversion
placement* matter? Because BF16 conversion is a memory op — it reads and
writes every element. Converting once instead of three times is three fewer
full traversals of the logits tensor, at every denoise step.

## Case 4: fused cast+pack into the latent

The denoised latent must be converted to BF16 and packed into the layout the
video VAE expects. In the naive layout that's a cast pass and a pack pass, two
full traversals of the latent (38.27 + 19.13 MiB at 512-class, 59.66 + 29.83
at 864-class). The fusion does cast+pack in one kernel, one traversal, and the
README's claim is the strongest kind: the fused output is **byte-identical**
to the unfused one — verified against the reference path. This is the fusion
equivalent of the "preserve the rounding boundary" rule below: it is legal to
change *when* bytes are converted; it is not legal to change *which* bytes
result.

## Case 5: SwiGLU inside threadgroup memory

The objective here is to keep the widest intermediate in the model — the
14336-wide SwiGLU product — entirely on-chip.

The MLP's SwiGLU (the gated activation: the elementwise product of a linear projection and the swish of another) computes two 14336-wide projections and a gated product. The
book's L16 instinct is to keep the intermediate on-chip. h3.c takes it
further: the whole SwiGLU tile — 32 KiB of threadgroup memory (the on-chip staging budget — sized so the tile stays resident rather than spilling to global memory) — is staged
inside the kernel and the gated product is computed in registers, so the
14336-wide intermediate **never leaves the SM** (streaming multiprocessor — the GPU's compute unit). The F32 accumulate happens
in registers; the result is rounded to BF16 exactly once, at the store. That
rounding placement is the actual contract:

??? question "Why does fusion have a rounding boundary?"
    Fusing changes *where* arithmetic happens, and IEEE rounding is not
    associative: `(a+b)+c` and `a+(b+c)` can differ in the last bit. A fused
    kernel that accumulates F32 and stores BF16 once can produce a *different
    bit pattern* than a chain of kernels that stores BF16 after every step.
    Neither is "more correct" — but if you fuse, you must decide *which*
    boundary you keep, and keep it everywhere. h3.c's rule, stated in code
    (`h3_gpu.h:289-290`): accumulate in F32, round to BF16 once at the store,
    never round in between. That single rule is what makes "byte-identical to
    the reference" a meaningful claim in 14g and 14h.
    [Full answer](qa.md#why-does-fusion-have-a-rounding-boundary)

## The counter-example: dead weight that proves the rule

Not every fusion wins. The GQA/MQA attention path (GQA — grouped-query attention, where a few key/value heads are shared across many query heads; MQA — multi-query, the extreme with a single key/value pair) was built as a cached
MPSGraph op (Metal Performance Shader Graphs — the compiler approach of
Lecture 21, Apple-style) and it is **still built, cached, and gated behind an
env var** because on the measured machines it's simply not better than the
hand-tuned kernels. The author keeps it because it costs nothing to ship
runtime-gated, and it's the fastest path on *some* future device. The lesson
isn't "never write MPSGraph" — it's the whole book's lesson: the second path
exists, it's measured, and the default is the number that won.

## Check yourself

1. Why does converting the head outputs to BF16 once, at load, beat converting
   them three times as each consumer reads them?
2. The cross-block AdaLN carry breaks at control-flow boundaries. Name two
   optimizations from this lecture series that would silently invalidate it,
   and how h3.c handles each.
3. The fused cast+pack claims byte-identical output. What single rule about
   rounding makes that claim *provable* rather than hopeful?

## Next

**[14e. Activation lifetimes](14e-activation-lifetimes.md)**: the other memory
budget — activations, and what reading their lifetimes buys.
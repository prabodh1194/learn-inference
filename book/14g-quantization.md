# 14g. Quantization with receipts

> Lecture 19 will teach you quantization mechanics. This lecture is about the
> *discipline* around it: h3.c's int8 path (int8 — 8-bit integers standing in for floats, with a scale that maps back) doesn't claim "2× smaller and
> roughly as good" — it ships with the design space named, the failure modes
> measured, and every tradeoff carrying a number. The receipt is the
> deliverable. The scale placement is the design.

## The contract

The objective here is to state exactly what gets quantized and what doesn't —
so the receipts later have a precise referent.

The DiT's linear layers run in int8 with BF16 (brain float16: 8-bit exponent, 7-bit mantissa) accumulation. The quantization
contract, in full:

- **Weights: one F32 (32-bit float) scale per output channel** (a scale is the multiplier that maps the stored integer back to the true value). Static, computed at load.
- **Activations: one F32 scale per row** (per token/patch), computed
  dynamically per kernel invocation.
- **Accumulate in F32, round once to BF16 at the store** — the 14d rounding
  boundary, unchanged.

The asymmetry is deliberate and it is the whole design. Weights are known at
load, so their scale can be per-channel and exact. Activations change every
step, so their scale is per-row and recomputed per invocation — but a row of
the grid is 5376 wide, and computing the row's max is a reduction over the
row, cheap relative to the matmul it precedes. What you *don't* do is use a
single global activation scale: a per-row scale is what keeps the dynamic
range of each patch honest, which is what a diffusion latent (the model's internal frame representation; per-patch
energy varying wildly across the frame) demands.

## The four FC2 strategies — the design space, mapped

The objective here is to map the quantization design space on the widest, most
numerically sensitive projection — and to show the choice is about math, not
speed.

The last projection of each MLP (FC2 — the MLP's second linear layer — 14336 → 5376) is the interesting case
because it's the widest and the most numerically sensitive — it feeds the
residual add, so its error propagates into the block's output directly. h3.c
implements **four** FC2 strategies (`h3_gpu.c`, "int8 fc2" and the fused
`fc2-f32` variant):

| Strategy | Weight scale | Product | Speed |
|---|---|---|---|
| `fc2-f32` | none (F32) | full-width F32 | baseline |
| `int8-fc2` | per-output-channel | full-width TensorOps (Apple's low-precision matrix-multiply path) int8 product | +3.1% |
| `int8-row-fc2` | per-output-channel + one scale per row | full-width TensorOps int8 product | +2.6% |
| `int8-mixed-fc2` | per-channel int8 + per-row, BF16 fallback | mixed, row-scaled | (measured, mixed) |

The winner isn't the fastest per kernel — it's the one that keeps the
*residual path* cleanest. The differences between int8-fc2 and int8-row-fc2
are sub-percent in speed; the difference in the *math* is that row scaling
changes *which* outputs carry rounding error. The choice is made with
**quality anchors attached to each**: SSIM (structural similarity — a 0–1 image-quality score, 1.0 being identical to the reference) 0.919 at 512-class and 0.828 at
864-class for the row-scaled path, and a documented "slightly less
numerically conservative" caveat for the int8 attention path — the README
says out loud that quantized attention can change *framing* (the 
conditioning's effect on composition) before it changes obvious artifacts.
Framing drift is the failure mode that SSIM misses; naming it is what makes
the tradeoff a decision instead of a surprise.

## The receipts, verbatim

The numbers the README attaches to the int8 path (upstream measurements, M5
Max, 768p, 24 fps):

```
forward time:    36.30 s  →  25.80 s  →  19.32 s      (F32 → int8 → int8 + fused)
peak storage:    36.4 GiB → 25.9 GiB →  (with streaming: ~2.0 GiB)
```

Only the DiT's linear layers are quantized — the encoders and decoders stay
BF16 — so the storage cut is a partial 29% (36.4 − 25.9 = 10.5 GiB saved), not
a full halving; and the ~2.0 GiB is 14c's streamed resident set applied on top.

The 2.6% row-fc2 speedup over plain fc2 (19.32 vs 19.85 s; (19.85 − 19.32) ÷ 19.85 = 2.7%) matters less than
what it buys in *composition*: the fused path combines the int8 matmul with
the F32 row-scale application, which changes the rounding boundary — and the
README's claim is that the fused result is **byte-identical to the unfused
int8 path**, verified against the reference. That claim is only meaningful
because 14d's rounding rule makes "byte-identical" a checkable predicate, and
because every variant is reachable at runtime via an env-var switch so the
comparison is A/B, not anecdote.

??? question "What does byte-identical mean, and why demand it?"
    Byte-identical means the fused kernel's output tensor is the same bit
    pattern as the unfused reference's, element for element — not "within
    1e-4". It's achievable here because the rounding rule (F32 accumulate,
    single BF16 store) is invariant under the fusion: the fusion changes *when*
    bytes are produced, not *which* bytes. It's the strongest claim you can
    make about an optimization, it costs nothing to verify (A/B run + compare),
    and it's the ceiling every other claim is measured against: if a fusion
    *isn't* byte-identical, you must explain the difference — that's the
    investigation 14h's approximation paths go through.
    [Full answer](qa.md#what-does-byte-identical-mean-and-why-demand-it)

## The meta-receipt: every switch is an A/B test

Every quantization variant, fusion, and aggressive path in 14h is
runtime-selectable (`--quantize`, env vars). This is not feature creep; it is
the measurement discipline of the whole book, shipped as code:

1. **The reference path is always runnable** — a regression can be pinned to
   the exact flag that caused it.
2. **The claim "2.6% faster" is reproducible by anyone with the binary** — not
   by trusting the README.
3. **A/B at scale**: the SSIM numbers and the L2 numbers (L2 — relative error vs. the reference, a Euclidean-distance ratio) of 14h are computed
   against the F32 reference, per variant, automatically (the `check_output`
   comparison harness in the test suite).

The design question for *your* engine, from Lecture 26: if your quantization
is a knob, what is your reference, and where does the comparison run?

## Check yourself

1. Why per-output-channel scales for weights but per-row scales for
   activations? What breaks if you swap them?
2. SSIM says "fine" for a quantized path that changes composition. What
   failure mode does the README name that SSIM can't see?
3. "Byte-identical to the reference" — what single rule about rounding makes
   that claim checkable, and what does it let you do with every other
   claim in the README?

## Next

**[14h. When optimization changes outputs](14h-aggressive-optimization.md)**:
token reduction, layer thinning, velocity reuse — the approximations, and the
oracle that keeps them honest.
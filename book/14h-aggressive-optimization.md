# 14h. When optimization changes outputs

> Everything so far was *lossless*: same bits out, fewer resources spent. The
> last level of the h3.c toolkit is different. Token reduction, layer
> thinning, velocity reuse, and core reuse all **change the output** — the
> video is measurably not the F32 reference's video. This is the hardest
> discipline in the engine, and the lectures to copy are not the tricks; they
> are the *oracle* that makes the tricks auditable, and the honesty about
> what changed.

## The stance: approximation is a product decision

Each approximation in h3.c has the same shape:

1. It is **opt-in** — an env var or flag turns it on; the reference path
   always exists.
2. It ships with **measured distance from the reference** — a number, per
   approximation, not "seems fine".
3. Its failure modes are **named in the README** — including the ones that
   look fine in metrics.

That stance is the whole lecture. An approximation without a measured distance
is not an optimization; it is a bug you haven't found yet.

## Token reduction: 28.3% faster, L2 says how wrong

The denoise loop's grid is ~2,835 rows at 512-class. Token reduction pools the
rows **2×2 in the spatial dimensions** and runs the attention over the pooled
grid — the block's *attention* sees 1/4 the rows, while the MLP and the
residual stream still run at full width. The measured cost: the loop's L2
(relative error vs. the reference latent) rises to **5.56% at 512-class and
15.14% at 864-class** — and the speedup is **28.3%** (39.13 s → 28.06 s at
768p), or 16.69 s → 12.60 s combined with int8.

The interesting numbers are the ones *not* reported as a single headline:
the L2 at 864-class (15.14%) is nearly 3× the 512-class error, and the README
does not hide it — the pattern at 864 gets more detail from fewer tokens, so
the error is worse exactly where the image is bigger. The honest summary in
the README names the visual consequences: less detail, less chromatic
consistency, and artifacts that were *tested* — the test found that
**thinning layers 40+ while reusing 3 middle layers caused chromatic ringing
with denoise steps ≥ 25** — so those two settings are not combined. That is
an optimization that was tried, failed in a documented way, and shipped with
the failure pinned in the code as the reason the combination is disallowed.
Repo rule: **keep the failure unedited** — it is the part that teaches.

## Layer thinning: the heuristic that failed once

Thinning skips whole blocks. The selection is by **gate rank**: the AdaLN
gates (14d) are per-block scalars measuring how much each block's output
contributes; the blocks whose gates are consistently smallest are the ones
skipped. It's a beautiful idea — the model itself tells you which blocks
matter.

Except block 1. The README notes the exception explicitly: block 1's gate is
small, but **skipping it breaks the output**. The measured reality: the first
block's function (building the residual stream's structure) matters in a way
the gate doesn't measure — the gate ranks *marginal contribution at its
position*, and early blocks have structural effects that are invisible to a
per-block scalar. The heuristic failed, and the code ships the exception
instead of the ideal: blocks 40+ thinnable (at 512-class), block 1
never. The lesson: **gate-ranked heuristics inherit the gate's blind spot**,
and a per-block scalar cannot see structure. The oracle (below) is what
caught it.

## Velocity reuse: extrapolation, clamped

The denoise loop predicts a velocity per step. Velocity reuse runs fewer
steps by **reusing a velocity from a previous step, extrapolated** — and the
extrapolation is **clamped to [−2, 2]** (the code enforces it). The clamp is
the interesting design decision: unclamped extrapolation can grow
unboundedly and drive the latent into the model's dead zones (the VAE's
domain is bounded); clamping caps the damage at a known amplitude. Every
approximation in this family has such a fence — a place where the author
decided what the worst case may be and enforced it in code, instead of
hoping.

## Core reuse: the algebraic approximation

The last one is the least documented — core reuse approximates the residual
blocks *algebraically* (reusing computed sub-expressions across blocks),
with the README's own verdict that the evidence is thinner: it's offered, it's
benchmarked as faster, and the author recommends it *only* in combination
with other savings at 512-class. This is the honest end of the approximation
spectrum: **an optimization can ship with weaker evidence, if the evidence
it does have is stated** — the failure is shipping it as if the evidence were
strong.

## The oracle pattern: keep the reference, measure the distance

Every approximation above is pinned by the same structure. The test suite has
an oracle: the F32 reference path, runnable at any time, whose outputs are
compared against every accelerated variant (the `test_output` / `check_output`
comparisons). Each variant's env-var switch is not just an A/B test — it's
the *restore path*: any artifact, any bug report, any "was it always like
this?" resolves in one command by running the reference.

??? question "What is an oracle, and why does every approximation need one?"
    An oracle is a reference implementation you trust more than the code you're
    testing. For h3.c it's the F32 path: slow, correct, always present. Every
    accelerated variant is compared against it automatically (relative L2,
    SSIM, byte-compare where lossless). Without an oracle, an approximation's
    claim is "I looked at the output and it seemed right" — with one, it's a
    number, computed on every change, catching drift like the layer-thinning
    gate failure the moment it happens. Your engine's oracle is the same
    shape: the un-optimized path, kept runnable, compared mechanically.
    [Full answer](qa.md#what-is-an-oracle-and-why-does-every-approximation-need-one)

## The layout bug: when "optimization" hides a corruption

The best story in the repo is a bug, not a trick. The checkpoint's QKV weight
rows are **interleaved per attention head** — for 56 heads, the rows for
head 0 of all queries, then head 1, etc. The naive reading (rows laid out
per-query, in order) *looks* correct: everything runs, nothing crashes, and
the output is **noisy but not garbage** — a corruption that reads like a
quality problem, not a correctness problem. It shipped, and it took an oracle
test (`test_bf16.c:1640-1674`) to catch: the test compares outputs against a
reference, and the mismatch surfaced the layout — fixed with a grouped-layout
flag (`checkpoint.h`), not by changing the model.

The lesson, stated as code: **a silent wrong-layout bug passes every
"does it run" test and fails only the "is it right" test.** If your engine
has no oracle, this bug is your engine too — you just haven't found it yet.
The fix also shows the right instinct: the layout is a *property of the
checkpoint*, so it's read from a flag in the checkpoint header, not assumed.

## The meta-lesson, and the honest summary

From the README's own closing words on these paths — each one is offered
with its cost measured, its failure modes named, and its escape hatch
shipped. The full list of what the aggressive paths buy, end to end:

```
F32 baseline:              39.13 s
+ token reduction:         28.06 s   (L2 5.56% / 15.14%)
+ int8 + fusions:          19.32 s
+ token reduction + int8:  12.60 s
```

...and the block-1 exception, the clamped extrapolation, the 864-class L2
honesty, and the chromatic-ringing disallowance are the *prices*, written
down. That is the deliverable. A faster engine that hides its prices is a
trap; a faster engine that publishes them is a tool.

## Check yourself

1. Block 1's gate is small, yet skipping it breaks the output. What does the
   gate actually measure, and what structural effect can't it see?
2. Velocity reuse clamps extrapolation to [−2, 2]. What is the clamp
   protecting, and what would unclamped growth do?
3. The QKV layout bug produced "noisy but not garbage" output. Which test in
   the suite caught it, and why would a kernel-level test have missed it?

## Next

**[15. Profiling](15-profiling.md)**: Part III starts here. The h3.c series
told you what a production engine *is*; profiling is how you find out where
*your* engine's time actually goes — on the NVIDIA box you're about to rent.
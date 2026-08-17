# 17. FlashAttention

**Build:** `kernels/triton/flash_attention.py` · **Test:** `tests/test_17_flash.py` (cuda)
**Moves:** attention time, and peak memory, from O(N²) to O(N)
**Prereq:** [16. Triton basics](16-triton-basics.md)

---

## The problem

Return to Lecture 02's arithmetic. Standard attention at N=4096, d=128:

```
memory  = 8N² + 8Nd bytes
compute = 4N²d + 3N² ops
intensity = 62 ops:byte     (vs. an H100's ridge of 295)
```

Memory-bound by nearly 5× on an H100, though on the 3090 you're actually
renting (ridge 76) it's memory-bound by only 1.2× (`76 / 62.4 = 1.22`), so
temper your speedup expectations accordingly. Note that both of those are
*this* N=4096 workload: the paper's own 3090 benchmarks show 2.5–4.5×
speedups at 4k–16k context (the A100 shows 2–4× in the same charts), because
there the `N²` traffic dominates and the 3090's lower bandwidth makes the
bytes you remove worth more. The roofline ratio predicts the small-N case;
the big-N case is the one FlashAttention was built for. And the `N²` terms dominate:
that's the score matrix `S = QK^T`, written to **HBM**, the GPU's large slow
memory, the fridge the chef walks to in Lecture 02's kitchen, and immediately
read back, twice:

```
1. S = QK^T          write S    (4096×4096 fp16 = 32 MiB)
2. P = softmax(S)    read S, write P
3. O = PV            read P
```

(fp16: half-precision numbers, 2 bytes each.) `S` is the grid of dot products
between every query and every key: "how much should this token listen to that
one", the whole per-head attention decision. It is written out in full, then
read back in full, then written out again as `P`, then read back again.

Four touches of the 32 MiB matrix, one per step of the bullet list: write S,
read S, write P, read P: **128 MiB of round-tripping to compute one attention
head** (one head: one parallel slice of the attention computation, the model
runs 16 of these side by side), on data that is never needed again (and the
same 8N² the Lecture 02 formula counts). It exists only because the algorithm
was written as three separate matrix operations.

Worse, memory is **quadratic** in sequence length. Doubling the context
doubles N, but the score matrix is N², so it quadruples. Check it on the
numbers:

```
N = 4096:   S is  4096² × 2 bytes  =  33,554,432 B  =   32 MiB
N = 8192:   S is  8192² × 2 bytes  =  134,217,728 B  =  128 MiB    4× the memory
```

and four touches of it, 4 × 128 MiB = **512 MiB of round-tripping per head at
N=8192**. That is what makes long context expensive: the scratch space, not the
math.

---

## The idea

FlashAttention (Dao et al., 2022) never materializes `S`. To materialize is to
write the whole thing out to memory, and the previous section's problem exists
only because standard attention insists on materializing both `S` and `P`.

The obstacle is softmax: it needs a sum over the whole row, so you seemingly can't
process K/V in pieces. The pieces are **tiles**: rectangular slices of a matrix,
small enough to fit in the chip's on-chip memory. The trick is that you *can*
process tiles, if you're willing to **fix up your answer as you go**.

### Online softmax

Numerically stable softmax subtracts the row max:

```
softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

Process a tile of K/V and you only know the max **so far**. When a later tile has a
larger value, everything computed already used the wrong max, but it's fixable:

```
m_new = max(m_old, m_tile)
correction = exp(m_old - m_new)          # rescale factor

acc = acc * correction + (new tile's contribution)
l   = l   * correction + (new tile's sum)
```

Keep a running max `m`, a running sum `l`, and a running output accumulator `acc`.
Each new tile rescales the accumulator by `exp(m_old - m_new)`. At the end, divide
by `l`.

**The result is exact.** Not an approximation, algebraically identical to standard
attention, computed in a different order. That's what makes FlashAttention safe to
use everywhere.

??? question "Does it really come out to the same denominator as one-shot softmax?"
    Run both on the same row and compare. Scores `[3, 1, 4, 1, 5, 2]`, tiles of
    two; `l` is the running sum of `exp(score − m)`:

    ```
    tile 1:  m = 3            l = e⁰ + e⁻²                        = 1.1353
    tile 2:  m = 4, move e⁻¹  l = 1.1353·e⁻¹ + e⁰ + e⁻³            = 0.4177 + 1.0498  = 1.4675
    tile 3:  m = 5, move e⁻¹  l = 1.4675·e⁻¹ + e⁰ + e⁻³            = 0.5398 + 1.0498  = 1.5896
    ```

    One shot, with the true max `5`:

    ```
    l = e⁻² + e⁻⁴ + e⁻¹ + e⁻⁴ + e⁰ + e⁻³ = 1.5896
    ```

    Identical. And the accumulator rescales by exactly the same `e^(m_old − m_new)`
    factor each step, so `acc / l` is the same weighted average the one-shot
    would compute — that's *why* the answer is exact, and why the rescaling
    steps are not optional decoration.
    [Full answer](qa.md#why-does-a-running-max-softmax-recompute-the-exact-same-denominator)

??? question "How can two orderings of floating-point math both be 'exact'?"
    "Exact" here means no approximation: every score, every weight, every sum
    is computed, in full, once. Nothing is truncated, sampled, or skipped, the
    way quantization (Lecture 19) or a genuine approximation would. Working
    row-in-pieces merely reorders the additions, and floating-point addition
    is not perfectly associative, so the last bits of rounding differ from
    standard attention, which is why the test tolerance is ~1e-2 rather than
    bit-for-bit equality. But there is no error beyond that rounding: the
    answer is as close as the same numbers in a different order can be.
    [Full answer](qa.md#how-can-two-orderings-of-floating-point-math-both-be-exact)

### Why it's faster

Q, K, and V tiles are loaded into **SRAM**, the chip's on-chip scratchpad:
~100 KB of shared memory per SM on the 3090 (a few MB across the whole chip),
about 20× faster to access than HBM, the big slow main memory whose ~936 GB/s
we measured in Lecture 04. The whole tile's work happens there, and only the
final output goes back to HBM.

Watch the terminology here, because two numbers get quoted for "on-chip
memory" and they are different things. The **caches** — ~128 KB of L1 per SM,
6 MB of L2 — are hardware-managed and invisible to your kernel. The
**scratchpad** FlashAttention uses is per-SM **shared memory** (also ~100 KB),
which the kernel allocates and controls explicitly (Lecture 20 shows the raw
CUDA). The bandwidth advantage over HBM is ~20× for the scratchpad, not the
"~100×" that sometimes gets quoted from cache numbers. Nothing in the
algorithm changes either way: tiles live where the kernel puts them, and the
traffic argument below is unaffected.

```
BEFORE:  read Q,K -> write S -> read S -> write P -> read P,V -> write O
AFTER:   read Q,K,V (in tiles) -> write O
```

Memory traffic drops from O(N²) to **O(N)**. Arithmetic is unchanged, slightly
increased, in fact, by the rescaling. **You do more math to move less data**, which
is exactly the right trade on a memory-bound operation.

This is the deepest lesson in Part III: on modern hardware, recomputation is often
cheaper than a memory round-trip.

---

## The structure

### First, the algorithm on its own

Before any Triton, here is the whole forward pass as the paper states it
(FlashAttention-1, Algorithm 1). No GPU vocabulary, just two loops and the
running state — read this until it makes sense, and the kernel afterwards is
only this with the loops rearranged:

```
init O=0, l=0, m=-inf                    # in HBM, one per query row-block

for j in 1..Tc:                          # outer: K,V column-blocks
    load K[j], V[j]  ->  SRAM

    for i in 1..Tr:                      # inner: Q row-blocks
        load Q[i], O[i], l[i], m[i]  ->  SRAM

        S  = Q[i] @ K[j].T               # [Br x Bc], on chip, never stored
        m~ = rowmax(S)                   # this tile's max
        P~ = exp(S - m~);  l~ = rowsum(P~)   # this tile's weights and their sum

        m_new = max(m[i], m~)                            # merge the maxes
        l_new = e^(m[i]-m_new)*l[i] + e^(m~-m_new)*l~    # merge the sums

        O[i]  = (1/l_new)*( l[i]*e^(m[i]-m_new)*O[i] + e^(m~-m_new)*P~@V[j] )

        store O[i], l[i]=l_new, m[i]=m_new  ->  HBM

return O    # exact softmax(QK^T)V — the N×N matrix was never built
```

**Two loops, and everything between the load and the store happens on chip.**
That is the entire kernel.

Read the update line by line, because each piece is doing one job:

- **`S = Q[i] @ K[j].T`** — the only place scores exist. `[Br × Bc]`, a *tile*
  of the score matrix, alive in SRAM for a few instructions and then gone. The
  `N × N` matrix from the opening section is never assembled anywhere.
- **`m~`, `P~`, `l~`** (the tilde means "for this tile alone") — a complete,
  correct softmax of the tile in isolation, using the tile's own max.
- **`m_new = max(m[i], m~)`** — the merge. Everything computed before this
  moment used the *old* max, so if this tile brought a bigger value, the history
  is now scaled wrong.
- **`e^(m[i]-m_new)` and `e^(m~-m_new)`** — the two correction factors, one for
  the history and one for this tile. Whichever max won, its factor is `e⁰ = 1`
  and the other shrinks. Nothing is ever scaled *up*, which is what keeps this
  numerically safe.
- **`l_new = ...`** — the same correction applied to the running denominator.
- **`O[i] = (1/l_new)*( l[i]*e^(...)*O[i] + e^(...)*P~@V[j] )`** — the
  accumulator. Note the `l[i]*` on the left: `O[i]` was stored *already divided*
  by the old `l[i]`, so you multiply it back out, add the new tile's
  contribution, and divide by the new total. That is why `l` must be stored
  next to `O` — you cannot undo the normalization without it.

Then the state goes back to HBM and the next `(i, j)` pair picks it up.

??? question "Why divide inside the loop instead of once at the end?"
    You can do either, and the Triton kernel below does it the other way —
    accumulate unnormalized, divide once after the loop. The paper's in-loop
    version keeps `O[i]` a *valid* attention output at every step, which is why
    it must store `l[i]` alongside it and multiply it back out each time. The
    accumulate-then-divide version skips that multiply-back and is slightly
    cheaper, which is why real kernels prefer it. The arithmetic is identical;
    it is only a question of where you put the division.

Now the same algorithm as a Triton kernel:

```python
@triton.jit
def flash_attention_kernel(Q, K, V, Out, softmax_scale, N, ...):
    # one block per (query tile, head)
    q_tile = tl.load(Q + ...)                   # stays in SRAM

    m_i = tl.full([BLOCK_M], -float("inf"))     # running max
    l_i = tl.zeros([BLOCK_M])                   # running sum
    acc = tl.zeros([BLOCK_M, HEAD_DIM])         # running output

    for start_n in range(0, N, BLOCK_N):
        k_tile = tl.load(K + ...)
        v_tile = tl.load(V + ...)

        s = tl.dot(q_tile, tl.trans(k_tile)) * softmax_scale
        # causal masking: a query never attends to a later key
        s = tl.where(causal_mask, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        p = tl.exp(s - m_new[:, None])
        correction = tl.exp(m_i - m_new)

        acc = acc * correction[:, None] + tl.dot(p, v_tile)
        l_i = l_i * correction + tl.sum(p, axis=1)
        m_i = m_new

    tl.store(Out + ..., acc / l_i[:, None])
```

**Compare the two loop orders.** The kernel above is not a transcription of the
algorithm — the loops have been turned inside out, and it is worth seeing
exactly what moved:

```
   FlashAttention-1 (the algorithm above)   FlashAttention-2 (the kernel above)
   K/V outer, Q inner                       Q outer, K/V inner

   for j in K/V blocks:        ← outer      for i in Q blocks:        ← outer
       load K[j], V[j]                          load Q[i]   (stays put)
       for i in Q blocks:      ← inner          m,l,acc = 0  (in registers)
           load Q[i],O[i],l[i],m[i]  ← HBM      for j in K/V blocks:  ← inner
           ...update...                             load K[j], V[j]
           store O[i],l[i],m[i]      → HBM          ...update...
                                                store O[i]            → HBM
                                                └─ ONCE, after the loop
```

Follow the running state. In FA-1 it lives in **HBM**: every `(i, j)` pair loads
`O[i], l[i], m[i]`, updates them, and writes them back — so a query block's
state crosses memory once per K/V block. In FA-2 the query block is the outer
loop, so `m`, `l`, and `acc` stay in **registers** for that block's entire walk
across the keys, and `O[i]` is written exactly once at the end.

Count the writes for one query block against `Tc` K/V blocks:

```
FA-1:  Tc loads + Tc stores of (O, l, m)
FA-2:   1 store of O
```

**Nothing in the math changed** — same online softmax, same corrections, same
exact result. What changed is which tile stays resident, and therefore how many
times the running state crosses HBM. That is the entire FA-1 → FA-2 improvement,
and it is the same lesson as the rest of the lecture: the arithmetic was never
the problem.

Your kernel is the FA-2 arrangement, which is what the official kernels use
today.

Three things that go wrong:

**Forgetting to rescale `acc`.** The most common bug. Output looks approximately
right, attention is a weighted average, so errors are subtle. It will pass a
casual eyeball and fail `allclose`. Trust the test.

**Causal masking at tile boundaries.** Tiles entirely above the diagonal can be
skipped; tiles on the diagonal need element-level masking. Getting this wrong
leaks future information: the model will look *better* at predicting, which is a
uniquely confusing bug.

**Wrong scale.** `1/sqrt(head_dim)`, applied before softmax.

??? question "Why divide by √d and not by d?"
    Scores are dot products, and a dot product of two `d`-dimensional vectors
    with unit-variance entries has variance `d`, not 1 — the `d` independent
    terms add their variances. So the scores scatter with standard deviation
    `√d`: at `d = 64` that's **8**, "±8", not the "±1" people guess. Between
    the most-liked and least-liked key of 64, the typical gap is ~2 sd ≈ 16,
    and what `exp` does with 16:

    ```
    e¹⁶ ≈ 8.9 million
    ```

    The whole row collapses onto one token. On the toy scores `[0, 6, 2, 2]`
    the top token grabs **96.2%** of the weight:

    ```
    e⁶ / (e⁶ + e⁰ + 2e²) = 403.4 / 419.2 = 96.2%
    ```

    A spiked softmax is nearly an argmax: attention "listens" to one key, and
    its gradient vanishes (a softmax row's Jacobian is `p(δ − p)`, ≈ 0 where
    one `p` is 1). Divide by `√d` first and the sd is back to 1, gaps of ~2,
    `e²` is a 7:1 ratio, and the same toy softmax stays soft — dividing `[0, 6,
    2, 2]` by √4 = 2 gives `[0, 3, 1, 1]`, top weight **76%**. The scalar keeps
    the *spread* of the scores under control, not the individual values.
    [Full answer](qa.md#why-divide-by-sqrtd-and-not-by-d)

**`-inf` minus `-inf` is NaN.** If a whole tile is masked out (entirely above the
diagonal) while `m_i` is still `-inf`, then `correction = exp(m_i - m_new)`
evaluates `exp(-inf - -inf)` = `exp(nan)` and poisons the accumulator. Skip
fully-masked tiles rather than masking them, or clamp `m_i` to a finite floor.
This one only bites on hardware, which is the worst place to find it.

---

## Build it

1. Implement the forward pass in `kernels/triton/flash_attention.py`. Inference
   only, no backward pass needed.
2. `uv run pytest tests/test_17_flash.py -v`, **`torch.allclose` against your
   Part II attention**, tolerance ~1e-2 for fp16.
3. Benchmark against PyTorch SDPA at N = 512 / 2048 / 8192. Report time **and
   peak memory**.
4. **Re-plot the roofline from Lecture 02** with your measured intensity. This is
   the payoff: you predicted 62 ops:byte for naive attention in Lecture 02, and
   now you measure what removing the round-trip actually bought.
5. Swap it into your engine; re-run the end-to-end benchmark.

---

## What you should see

**Speedup growing with sequence length.** Small at N=512, large at N=8192, the
`N²` term you removed only dominates when N is big.

**Peak memory much lower**, and now linear in N rather than quadratic. Often this
matters more than the speed: it's what makes long context feasible at all.

**Higher measured arithmetic intensity**: you moved right along the roofline.

**You will probably not beat the official FlashAttention.** It's hand-tuned per
architecture with warp specialization (different warps, the chip's fixed 32-thread
groups, specialise on different jobs: some fetch the next tile while others
compute) and careful pipelining (fetching the next tile overlaps with the
arithmetic of the current one, so the memory never idles). Getting within 2× of
it in ~100 lines of Triton is a genuinely good result, and knowing *why* the gap
exists is the point.

---

## Go deeper

- **[FlashAttention](https://arxiv.org/abs/2205.14135)** (Dao et al., 2022),   read §3.1 for the tiling algorithm and Algorithm 1. You've now implemented it,
  so the paper reads as confirmation rather than instruction.

  Two things in the abstract are worth reading closely. It calls the algorithm
  **"IO-aware"**: the framing that memory movement between HBM and SRAM, not
  FLOPs, is the quantity to optimize. That's Lecture 02's roofline, stated as a
  design principle.

  And it proves the IO complexity is **optimal for a range of SRAM sizes**, not
  merely better, but provably the least HBM traffic possible. That's rare in
  systems work, and it's why FlashAttention became the default rather than one
  option among several.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)**: better work
  partitioning; explains where your version's remaining gap comes from.
- **[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)**
  (Milakov & Gimelshein): the running-max trick in isolation. Short and clear.
- **Kiely §2.5** (p.67–70), FlashAttention and PagedAttention as the two
  attention optimizations, now with your own kernel as the reference point.
- **[Triton tutorial 06](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)**:
  a reference implementation. Try yours first.

---

## Check yourself

1. FlashAttention does *more* arithmetic than standard attention and is faster.
   Explain, in terms of Lecture 02.
2. Why is the result exact rather than approximate?
3. Speedup is 1.2× at N=512 and 6× at N=8192. Why does it scale with N?
4. What does forgetting to rescale `acc` do to the output, and why won't you catch
   it by looking?
5. Compare your measured intensity to Lecture 02's predicted 62. Did it move as
   much as you expected?

---

## Next

**[18. A paged attention kernel](18-paged-attention-kernel.md)**: combine this
with L09's block tables.

Start from the kernel you just wrote and change **only** the K/V addressing to
go through the block table. Nothing else about the algorithm moves.

This is where paging stops costing you latency, you keep L09's memory win and
give back the per-step penalty.

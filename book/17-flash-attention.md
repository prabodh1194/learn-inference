# 17 — FlashAttention

**Build:** `kernels/triton/flash_attention.py` · **Test:** `tests/test_17_flash.py` (cuda)
**Moves:** attention time, and peak memory — from O(N²) to O(N)
**Prereq:** [16 — Triton basics](16-triton-basics.md)

---

## The problem

Return to Lecture 02's arithmetic. Standard attention at N=4096, d=128:

```
memory  = 8N² + 8Nd bytes
compute = 4N²d + 3N² ops
intensity = 62 ops:byte     (vs. an H100's ridge of 295)
```

Memory-bound by nearly 5×. And the `N²` terms dominate — that's the score matrix
`S = QK^T`, written to HBM and immediately read back, twice:

```
1. S = QK^T          write S    (4096×4096 fp16 = 32 MiB)
2. P = softmax(S)    read S, write P
3. O = PV            read P
```

**64 MiB of round-tripping to compute one attention head**, on data that is never
needed again. It exists only because the algorithm was written as three separate
matrix operations.

Worse, memory is **quadratic** in sequence length. Doubling context quadruples the
scratch space, which is what makes long context expensive.

---

## The idea

FlashAttention (Dao et al., 2022) never materializes `S`.

The obstacle is softmax: it needs a sum over the whole row, so you seemingly can't
process K/V in pieces. The trick is that you can — if you're willing to **fix up
your answer as you go**.

### Online softmax

Numerically stable softmax subtracts the row max:

```
softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))
```

Process a tile of K/V and you only know the max **so far**. When a later tile has a
larger value, everything computed already used the wrong max — but it's fixable:

```
m_new = max(m_old, m_tile)
correction = exp(m_old - m_new)          # rescale factor

acc = acc * correction + (new tile's contribution)
l   = l   * correction + (new tile's sum)
```

Keep a running max `m`, a running sum `l`, and a running output accumulator `acc`.
Each new tile rescales the accumulator by `exp(m_old - m_new)`. At the end, divide
by `l`.

**The result is exact.** Not an approximation — algebraically identical to standard
attention, computed in a different order. That's what makes FlashAttention safe to
use everywhere.

### Why it's faster

Q, K, and V tiles are loaded into **SRAM** (on-chip, ~100× faster than HBM), the
whole tile's work happens there, and only the final output goes back to HBM.

```
BEFORE:  read Q,K -> write S -> read S -> write P -> read P,V -> write O
AFTER:   read Q,K,V (in tiles) -> write O
```

Memory traffic drops from O(N²) to **O(N)**. Arithmetic is unchanged — slightly
increased, in fact, by the rescaling. **You do more math to move less data**, which
is exactly the right trade on a memory-bound operation.

This is the deepest lesson in Part III: on modern hardware, recomputation is often
cheaper than a memory round-trip.

---

## The structure

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

Three things that go wrong:

**Forgetting to rescale `acc`.** The most common bug. Output looks approximately
right — attention is a weighted average, so errors are subtle. It will pass a
casual eyeball and fail `allclose`. Trust the test.

**Causal masking at tile boundaries.** Tiles entirely above the diagonal can be
skipped; tiles on the diagonal need element-level masking. Getting this wrong
leaks future information — the model will look *better* at predicting, which is a
uniquely confusing bug.

**Wrong scale.** `1/sqrt(head_dim)`, applied before softmax.

---

## Build it

1. Implement the forward pass in `kernels/triton/flash_attention.py`. Inference
   only — no backward pass needed.
2. `uv run pytest tests/test_17_flash.py -v` — **`torch.allclose` against your
   Part II attention**, tolerance ~1e-2 for fp16.
3. Benchmark against PyTorch SDPA at N = 512 / 2048 / 8192. Report time **and
   peak memory**.
4. **Re-plot the roofline from Lecture 02** with your measured intensity. This is
   the payoff: you predicted 62 ops:byte for naive attention in Lecture 02, and
   now you measure what removing the round-trip actually bought.
5. Swap it into your engine; re-run the end-to-end benchmark.

---

## What you should see

**Speedup growing with sequence length.** Small at N=512, large at N=8192 — the
`N²` term you removed only dominates when N is big.

**Peak memory much lower**, and now linear in N rather than quadratic. Often this
matters more than the speed: it's what makes long context feasible at all.

**Higher measured arithmetic intensity** — you moved right along the roofline.

**You will probably not beat the official FlashAttention.** It's hand-tuned per
architecture with warp specialization and careful pipelining. Getting within 2× of
it in ~100 lines of Triton is a genuinely good result, and knowing *why* the gap
exists is the point.

---

## Go deeper

- **[FlashAttention](https://arxiv.org/abs/2205.14135)** (Dao et al., 2022) —
  read §3.1 for the tiling algorithm and Algorithm 1. You've now implemented it,
  so the paper reads as confirmation rather than instruction.
- **[FlashAttention-2](https://arxiv.org/abs/2307.08691)** — better work
  partitioning; explains where your version's remaining gap comes from.
- **[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)**
  (Milakov & Gimelshein) — the running-max trick in isolation. Short and clear.
- **Kiely §2.5** (p.67–70) — FlashAttention and PagedAttention as the two
  attention optimizations, now with your own kernel as the reference point.
- **[Triton tutorial 06](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)**
  — a reference implementation. Try yours first.

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

**Next:** [18 — A paged attention kernel](18-paged-attention-kernel.md) — combine
this with Lecture 09's block tables.

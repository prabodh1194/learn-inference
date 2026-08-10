# Q&A: worked answers

Real questions asked while working through this book, with the answers that
clarified them. **Several of these caught genuine errors in the lectures.**

Kept because the confusions are reproducible, if one tripped a careful reader
once, it will trip the next one. Each links to the lecture it belongs to.

!!! tip "Add to this file"
    When something in the book confuses you and the explanation lands, write it
    here. The misunderstanding is the valuable part, not the answer.

---

## What's an attention head? How is it different from a KV head?

**Lecture:** [01. The two phases](01-the-two-phases.md) · [05. The KV cache](05-kv-cache.md)

An attention head asks: *for this token, which earlier tokens should I look at,
and what should I take from them?* It does that with three projections:

- **Q (query)**: what I'm looking for
- **K (key)**: what I offer, so others can find me
- **V (value)**: what I actually hand over

It computes `softmax(Q·Kᵀ)·V`. Models have many heads per layer so they can
specialize, one tracks syntax, another the sentence subject, another matches
quotation marks.

Qwen3-0.6B has **16 attention heads but only 8 KV heads**. Classic Multi-Head
Attention ties them 1:1; **Grouped-Query Attention (GQA)** lets pairs of queries
share one K/V:

```
MHA (16 KV heads)          GQA (8 KV heads)
Q0 → K0,V0                 Q0 ┐
Q1 → K1,V1                 Q1 ┴→ K0,V0
Q2 → K2,V2                 Q2 ┐
...                        Q3 ┴→ K1,V1
16 separate K/V            8 shared K/V
```

**Why it's in this book:** only K and V get cached. Q is recomputed each step
from the current token, you never need a past query again. So the cache is
sized by *KV* heads:

```
112 KiB/token = 2 (K,V) × 28 layers × 8 KV heads × 128 head_dim × 2 bytes
                                      ^^^^^^^^^^ 8, not 16
```

With 16 KV heads that's 224 KiB/token, double. Decode is memory-bound, so GQA
is a direct 2× on the exact thing that bottlenecks you.

---

## Is every token 2 bytes?

**Lecture:** [02. Arithmetic intensity](02-arithmetic-intensity.md)

No, **2 bytes is one number**, not one token. fp16 is 16 bits.

A token id is an integer, but after embedding it becomes a *vector*. Its cached
state is ~57,000 numbers:

```
2 (K,V) × 28 layers × 8 KV heads × 128 head_dim = 57,344 numbers
× 2 bytes each                                  = 112 KiB
```

The "2 bytes" is a **precision choice**, and it's exactly what
[Lecture 19](19-quantization.md) attacks: go to int8 and every number is 1 byte,
halving both the cache and the weight traffic.

??? note "So how many matmuls per token?"
    Seven per layer (Q, K, V, O, and three MLP matrices (gate, up, down)) so
    **196 matmuls** across 28 layers, plus the `Q·Kᵀ` and `·V` products inside
    attention.

    Heads do *not* each get their own matmul. All 16 heads come from **one**
    1024→2048 projection which is then *reshaped* into 16 chunks of 128. Heads
    are a view of one tensor, not 16 operations.

---

## What is the ops:byte math actually for?

**Lecture:** [02. Arithmetic intensity](02-arithmetic-intensity.md)

One division that answers: **which of the GPU's two limits am I hitting?**

A GPU does two things at fixed maximum rates, arithmetic (~71 TFLOP/s on a
3090) and moving bytes (~936 GB/s). Divide them and you get what the *hardware*
needs: **76 operations per byte** to keep its compute units fed. Then compute the
same ratio for your *algorithm* and compare.

From the demo:

| | ops:byte | vs 3090's 76 | verdict |
|---|---|---|---|
| prefill | 512 | above | **compute-bound** |
| decode | 0.92 | far below | **memory-bound** |

The value is in what it rules out. Decode at 0.92 uses ~1% of the GPU's
arithmetic, so a faster GPU buys you nothing; more bandwidth, fewer bytes, or
more work per byte are the only levers.

!!! warning "You are not meant to memorize this"
    Lecture 02 quotes Kiely on exactly this point: computing arithmetic intensity
    by hand is *"an academic exercise, not a routine task for inference
    engineers."* Do it once to build the intuition, then let
    `book/code/roofline.py` do it.

    The three sentences worth retaining: **prefill is compute-bound; decode is
    memory-bound; which one you're in depends on how many tokens share one weight
    load.**

---

## Why do prefill and decode have different weight requirements?

**Lecture:** [01. The two phases](01-the-two-phases.md)

**They don't.** Both read all 840 MiB. The difference is *how many tokens share
one read*, prefill amortizes across 512, decode across 1.

The question usually comes from conflating two different things:

| | Weights (840 MiB) | KV cache (112 KiB/token) |
|---|---|---|
| What | learned parameters | K,V for each token |
| Changes per request? | never | grows every token |
| Prefill | reads all, **once** | *writes* 512 tokens' worth |
| Decode | reads all, **per token** | *reads* all, appends one |

??? note "But doesn't prefill build KV incrementally too?"
    Yes, prefill writes K,V for all 512 prompt tokens (~56 MiB). But compare
    the scale:

    ```
    prefill:  840 MiB weights (once) +  56 MiB KV written  = ~900 MiB
    decode:   840 MiB × 256 tokens   + growing KV re-read  = ~233,000 MiB
    ```

    Prefill's KV write is ~6% of its traffic. Decode's repeated weight reads are
    ~99% of its.

??? warning "Where that flips, long context"
    Past ~8k tokens, decode's KV re-read overtakes the weight read:

    | Context | Weights/step | KV read/step |
    |---|---|---|
    | 512 | 840 MiB | 56 MiB |
    | 8k | 840 MiB | 896 MiB, **equal** |
    | 32k | 840 MiB | 3,584 MiB, **KV dominates** |

    This is why long-context serving is a different engineering problem, and why
    KV-cache quantization is a separate lever from weight quantization.

---

## Does prefill build a matrix of growing prefixes?

**Lecture:** [01. The two phases](01-the-two-phases.md)

A natural guess, and the fix explains why prefill is one shot:

```
guess                     actual
c                         [cat]     <- token 0
ca                        [ sat]    <- token 1
cat                       [ on]     <- token 2
cat s                     [ mat]    <- token 3
```

**One row per token**, each row that token's embedding, a `(4, 1024)` matrix.
Not a growing prefix per row.

The "attends only to what came before" property comes from the **causal mask**,
not the input layout. All 4 tokens project to Q/K/V at once, attention computes a
4×4 score matrix, then masks the upper triangle:

```
        cat   sat   on    mat
cat  [   ✓    -∞    -∞    -∞  ]
sat  [   ✓    ✓     -∞    -∞  ]
on   [   ✓    ✓     ✓     -∞  ]
mat  [   ✓    ✓     ✓     ✓   ]
```

The prefix intuition is *exactly* what that triangle encodes, but expressed as
a mask on one big matmul rather than as separate padded inputs. **That's why
prefill needs no sequential loop.**

Decode has one row, so `Q` is `(1, 1024)`, a vector. Same weights, same kernel,
1 row instead of 512. That *is* the 512 vs 0.92 ops:byte.

---

## Can batches be larger than the sequence length?

**Lecture:** [07. Static batching](07-static-batching.md) · [09. Paged attention](09-paged-attention.md)

Yes: the tensor is 3-D, `(batch, tokens, hidden)`, and **both** axes raise
ops:byte by putting more rows behind one weight load.

| | grow tokens | grow batch |
|---|---|---|
| Called | longer prompt | batching |
| Whose choice | the user's | **yours** |
| Costs | quadratic attention | linear KV cache |

The second row is why batching is the lever the field pulls: prompt length
arrives with the request, batch size is yours to set.

**What stops you going to 512:** capacity, not bandwidth.

```
112 KiB/token × 512 tokens ×  32 seqs =  1.75 GiB   ✓
112 KiB/token × 512 tokens × 512 seqs = 28 GiB      ✗ on a 24GB card
```

Weights stay 840 MiB regardless. **The KV cache is what scales with batch, and
what runs out.** Hence [Lecture 09](09-paged-attention.md).

??? note "And the waste that motivates Lecture 08"
    Real requests differ in length but a tensor is rectangular, so you pad, and
    the GPU computes those PAD columns and discards them. On realistic mixed
    traffic that's **61% waste** (`batching_waste.py`). Lecture 08 fixes it by
    scheduling per step instead of padding into a fixed rectangle.

---

## Why is decode memory-bound if the weights are already on the GPU?

**Lecture:** [00. Introduction](00-intro.md)

Because "on the GPU" means **in VRAM**, and VRAM is not where arithmetic happens.

The weights are copied to VRAM once at startup. The traffic that costs you is
VRAM → the GPU's on-chip SRAM and registers, and it repeats **every forward
pass**:

| Path | Bandwidth (3090) | When |
|---|---|---|
| **VRAM → on-chip** | ~936 GB/s | **every step**, the bottleneck |
| CPU RAM → VRAM (PCIe) | ~64 GB/s | once at load time |

??? question "Why can't the GPU just keep the weights on-chip?"
    **SRAM is tiny.** A 3090 has ~128 KB of L1 per SM and 6 MB of L2, against
    840 MiB of weights, off by ~140× even against L2. A weight streams in, gets
    used once, and is evicted before reuse.

    That's why decode's ceiling is a *bandwidth* number, not a cache-hit-rate
    number. It's also the constraint FlashAttention exploits deliberately
    ([Lecture 17](17-flash-attention.md)): the 32 MiB score matrix doesn't fit
    either, so tile the computation until each tile does.

    If weights genuinely crossed PCIe each step you'd be ~15× slower again. That
    case exists: it's called **offloading**, and it's a last resort for models
    that don't fit.

---

## Where does 232,960 MiB come from?

**Lecture:** [01. The two phases](01-the-two-phases.md)

The demo's decode row, for a 512-token prompt and **256 tokens out**. Almost all
of it is one multiplication:

```
840 MiB   all the weights, re-read to produce ONE token
x 256     tokens generated
= 215,040 MiB
```

Plus the KV cache, which each step also re-reads, and which grows as you go:

```
context starts at 512, ends at 768, averages ~640 tokens
640 x 112 KiB x 256 steps ~= 17,920 MiB
```

Total **~232,960 MiB**, about 228 GiB moved to produce 256 tokens. Against
prefill's 840 MiB that is **277x the traffic for half the arithmetic**, because
prefill reads the weights once for 512 tokens while decode reads them 256 times
for 256 tokens.

??? note "Sanity-check it against the hardware"
    This is exactly why decode is slow, and it predicts the speed:

    ```
    232,960 MiB / 936 GB/s = 0.26 s  ->  ~980 tokens/sec
    ```

    That is a single stream's theoretical ceiling on a 3090, set purely by
    bandwidth. Real engines land well under it. If you ever measure *above* it,
    something is wrong with your measurement.

!!! tip "What to actually remember"
    Not 232,960. The shape:

    **decode traffic ~= weights x tokens generated**

    The KV term is ~8% here and ignorable at short context. Past ~8k tokens it
    takes over, which is a different engineering problem (see
    [the long-context note](#why-do-prefill-and-decode-have-different-weight-requirements)).

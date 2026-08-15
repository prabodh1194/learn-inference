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

## You double the FLOPS but keep the bandwidth: what happens to TTFT and TPS?

**Lecture:** [01. The two phases](01-the-two-phases.md) · [02. Arithmetic intensity](02-arithmetic-intensity.md)

**Wrong intuition:** "TPS is a compute number, so more compute must speed it up."
It isn't. TPS (1/TPOT) is set by decode, and decode is memory-bound: one token's
time is *bytes moved / bandwidth*, and FLOPS appears nowhere in that ratio.

| | double FLOPS | double bandwidth |
|---|---|---|
| **TTFT** (prefill, compute-bound) | roughly **halves** | unchanged* |
| **TPS** (decode, memory-bound) | **unchanged** | roughly **doubles** |

TTFT is dominated by prefill, which is compute-bound:

```
TTFT ≈ 2 × params × prompt_tokens / achieved_FLOPS
```

Double the FLOPS, halve the prefill time. Two caveats:

- "achieved", not the spec sheet's peak, and only while prefill is genuinely
  compute-bound. Short prompts are launch- and tokenizer-bound, and TTFT also
  includes queue wait, so tiny prompts won't halve.
- The same change leaves decode exactly where it was. The GPU loads the same
  bytes per token; how fast it could multiply them is irrelevant.

The rule that generalizes: **TPS moves only when arithmetic intensity moves**;
more tokens sharing one weight load (batching, longer prompts, prefix reuse).
Peak FLOPS is not on that list. *Renting a faster GPU fixes TTFT; it does not
fix decode.*

\* Double the bandwidth and prefill barely moves: at a 512-token prompt its
intensity is ~511 ops:byte, far above the 3090's ridge of ~76, so it stays
compute-bound. Bandwidth only starts to matter there at very short prompts.

---

## What happens when the batch gets too big?

**Lecture:** [01. The two phases](01-the-two-phases.md) · [07. Static batching](07-static-batching.md)

**Wrong intuition:** "A big enough batch hits the FLOPS limit and then starts
getting *slower*."

It saturates; it doesn't reverse. While decode is memory-bound, the per-step
cost is the weight load, which is fixed; batch 32 costs almost the same as
batch 1. Once compute takes over, per-step time grows linearly with batch, so
throughput (batch ÷ step time) plateaus. Where? One output token costs
`2 × params` FLOPs (one multiply-accumulate per weight), so a step of batch B
costs `2 × params × B`, and:

```
throughput  =  B / step_time  =  B / (2 × params × B / peak_FLOPS)
            =  peak_FLOPS / (2 × params)        <- independent of B
```

More batch after that buys nothing and adds latency to every request in it.

The crossover is computable, and for this model it mostly doesn't exist. Per
step, weight bytes are fixed at `2 × params`; KV bytes are
`kv_per_token × context × batch`:

```
intensity(B) = 2·params·B / (2·params + kv_per_token·C·B)   →   ~7,700/C ops:byte
```

As batch grows, intensity approaches a ceiling of about **7,700 ÷ context** for
Qwen3-0.6B (attention FLOPS add a small +2). Against the 3090's ridge of ~76:

- context 768 (typical chat): ceiling ≈ 12; memory-bound at **every** batch.
- context 64: ceiling ≈ 122; compute-bound only past roughly B≈200.

So the FLOPS wall is a short-context curiosity here. What breaks first is the
KV cache: at a few thousand tokens of context, a 24 GB card runs out of blocks
before decode ever runs out of arithmetic. That is why [Lecture 09](09-paged-attention.md)
answers "what eventually breaks" with *memory*, not FLOPS.

[Also: can batches be larger than the sequence length?](#can-batches-be-larger-than-the-sequence-length)

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
    **SRAM is tiny.** The scratchpad arithmetic actually runs in is the
    shared memory attached to each SM: ~100 KB per SM on the 3090. Pooled
    across all 82 SMs that's ~8.2 MB — and even that pooled total can't hold
    a working set, because a tensor lives on **one** SM at a time — against
    840 MiB (880.8 MB) of weights:

    ```
    880.8 MB / ~8.2 MB  =  ~107×
    ```

    (The 6 MB of L2 and 128 KB of L1 per SM that get quoted for "on-chip
    memory" are *caches*, and are not what FlashAttention's tiles occupy.
    The scratchpad is per-SM shared memory, and its bandwidth advantage over
    HBM is ~20×, not the ~100× sometimes claimed.)

    A weight streams in, gets used once, and is evicted before reuse.

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

---

## Aren't Q, K, and V pre-calculated? Aren't we just loading them during inference?

**Lecture:** [05. The KV cache](05-kv-cache.md)

No, and this is a load-bearing distinction. Three different things travel
through the name "Q/K/V":

1. **The weight matrices W_Q, W_K, W_V**: pre-calculated (trained once), fixed,
   and only **loaded** during inference. Loading these is the 840 MiB of pure
   memory traffic that makes decode memory-bound.
2. **The per-token K/V vectors**: *not* pre-calculated. They are activations,
   produced at runtime by multiplying each token's hidden state against the
   weights: `K = X·W_K`. This is a matrix multiplication, real arithmetic, and
   it happens on the GPU while the model runs.
3. **Q**: an activation too, recomputed every step and thrown away (never
   cached, because the query changes every step).

So a decode step both **computes and loads**:

```
compute:  Q for the current token        (used once, discarded)
          K and V for the current token  (computed once, appended to the cache)
load:     all 840 MiB of weights         (the memory-bound part)
          every past K/V from the cache  (grows with context)
```

The cache is a *store* for step 2's output. Each token's K/V enters the cache
once, during prefill for prompt tokens and during decode for generated ones,
and from then on is only loaded. The "computed once, loaded many times" pattern is
the whole point.

**The wrong intuition, and why it's wrong:** "pre-calculated" is true of the
*weights*. The K/V vectors can't be pre-calculated because they depend on the
input tokens, which don't exist until the request arrives. If K/V were only
loaded, prefill would have nothing to fill the cache with, and Lecture 03's
99.6%-waste demo (163,584 K/V vectors *computed* to use 512 of them) would be
meaningless.

The confusion is worth having explicitly, because the same sentence "we just
load it" is correct for the weights and wrong for the cache. Decode is
memory-bound because of the **loads** (weights + past K/V), not because it
stopped computing the new K/V. The new token's K/V projection is a small
matmul; the load that dominates is the one that scales with context.

---

## Why does the repetition penalty divide positive logits but multiply negative ones?

**Lecture:** [06. Sampling](06-sampling.md)

**Wrong intuition:** "A penalty should shrink every score, so just divide all the
generated tokens' logits by the penalty."

The flaw: dividing a **negative** number moves it **toward zero**, and closer to
zero means *more* probable, not less. Division is a penalty for positive logits
and a reward for negative ones.

The penalty's real job is to push each penalized logit **away from zero**. A
logit between -1 and 1 is a coin flip; the farther from zero it is, the more the
model commits to favor (positive) or reject (negative) the token. So:

```
penalty = 1.2, token "cat" was already generated:

  logit +5.0  ->  +5.0 / 1.2 = +4.17   pushed down,   less likely   ✓
  logit -3.0  ->  -3.0 × 1.2 = -3.6    pushed down,   less likely   ✓
  naive:      ->  -3.0 / 1.2 = -2.5    pushed UP toward zero, MORE likely  ✗
```

A concrete two-token example. After "cat" was generated, the model's raw logits
are: cat = 1.0, dog = 0.8. Without any penalty, softmax gives:

```
P(cat) = e^1.0 / (e^1.0 + e^0.8) = 2.718 / (2.718 + 2.225)  =  55.0%
```

With penalty 1.2 applied to "cat":

```
cat  =  1.0 / 1.2  =  0.833
P(cat) = e^0.833 / (e^0.833 + e^0.8) = 2.300 / (2.300 + 2.225)  =  50.8%
```

"cat" fell from 55% to 51%, and "dog" picked up the difference. Now flip the
signs: cat = -1.0, dog = 0.8 (the model mildly dislikes "cat" already). The
naive division gives cat = -0.83, which is *closer* to dog's neighborhood than
before:

```
P(cat), penalty off:     e^-1.0 / (e^-1.0 + e^0.8)  =  0.368 / 2.594  =  14.2%
P(cat), naive ÷1.2:      e^-0.83 / ...              =  0.436 / 2.662  =  16.4%   (up!)
P(cat), correct ×1.2:    e^-1.2 / ...               =  0.301 / 2.527  =  11.9%   (down ✓)
```

The naive version makes a previously generated token *more* likely precisely
when the model was already leaning against it, which is the opposite of what the
parameter promises. That's why the correct form branches on the sign, and why
"sign matters!" sits in the lecture's code.

One more edge case: a logit of exactly zero is untouched by either operation
(0/1.2 = 0×1.2 = 0), which is correct, a token the model is indifferent to
shouldn't be moved at all by the sign convention.

---

## Why is temperature 0 special-cased instead of a very small divisor?

**Lecture:** [06. Sampling](06-sampling.md)

**Wrong intuition:** "T=0.0001 is basically greedy, so why does the code branch
to argmax?"

Two reasons, one mechanical and one principled.

**Mechanical.** `T=0` is division by zero. `logits / 0.0` produces NaN on CUDA,
and `softmax` of a vector full of NaN is NaN; the sample comes back garbage.
So the code must branch on `T == 0` regardless.

**Principled.** "Basically greedy" is not greedy. With `T = 1e-9` you still
have a proper probability distribution over the whole vocabulary; the
second-best token's probability is not zero, it's `e^-δ/T` for the logit gap
`δ`, which underflows to ~1e-434 at δ=1. The sampler *can't* pick token #2 in
practice, but the path is probabilistic, and more importantly the guarantee is
not structural: any future change (a tie between two max logits, a float
rounding, a fused kernel) can tip it.

Greedy-as-argmax is this lecture's test oracle: every later optimization
(batching, paging, prefix caching, speculative decoding) is asserted to produce
*exactly* the same tokens as the reference. That assertion only has teeth if
the greedy path is deterministic by construction, not by "the odds are
astronomically against a different token." So `T=0` becomes its own mode with
a guarantee, which is also what makes run-to-run output differences
unambiguous to debug (check-yourself Q3).

---

## Where do the 61% and 64% waste numbers in Lecture 07 come from?

**Lecture:** [07. Static batching](07-static-batching.md)

**Wrong intuition:** "Those must be measured on a GPU, or at least from real
generation runs."

They're the output of a **simulation** (`book/code/batching_waste.py`) that
counts slots; it never runs a model. The workload is 32 requests drawn with
`random.seed(0)`: prompts of 16–512 words, outputs of 8–512 tokens (word
count stands in for token count). The counting rules are exactly the
definition of static batching:

```
prefill:  each batch pads every prompt to its longest member
          slots_per_batch  =  longest_prompt × 8
          useful           =  the actual prompt tokens

decode:   every slot stays occupied until the batch's longest output ends
          slots_per_batch  =  longest_output × 8
          useful           =  the actual output tokens
```

Run the tally by hand on batch 0 (prompts 128, 16, 256, 128, 32, 256, 512,
256; outputs 16, 512, 128, 512, 8, 256, 128, 16):

```
prefill:  longest prompt = 512  ->  512 × 8 = 4,096 slots, 1,584 useful
          waste = 1 - 1,584/4,096 = 61.3%
decode:   longest output = 512   ->  512 × 8 = 4,096 slots, 1,576 useful
          waste = 1 - 1,576/4,096 = 61.5%

all 4 batches:  prefill 16,384 slots / 5,904 useful  =  64.0%
                decode  14,336 slots / 5,584 useful  =  61.0%
```

So "61%": 61 out of every 100 decode slot-steps in the simulation are a
finished request sitting in a held slot. "64%": 64 of every 100 padded prompt
slots are filler. The 2.57× is the same tally as a ratio
(`14,336 / 5,584 = 2.57`).

Two honest caveats so the numbers stay fair. First, they're word counts, not
token counts, and approximate; the real `batch_bench.py` run on the actual
model will print different (and larger) numbers. Second, the demo is the
*ideal* continuous batching (zero overhead, always work waiting), which is why
the book calls 2.57× a ceiling you'll measure under.

---

## Why can't the sequence's cache just grow as it needs space?

**Lecture:** [09. Paged attention](09-paged-attention.md)

The natural reaction to the book's reservation numbers is: why book 32,768
columns for a chat that will use 128? Why not give each conversation a small
cache and grow it on demand, like Python lists do?

The wrong intuition is that memory is a free-for-all you can carve anywhere.
On the GPU the cache for one sequence must be one **continuous strip**: the
attention kernel reads the whole strip as a single block in one read. A strip
can only grow at its end, into the space immediately after it:

```
row A:  [  used, 128 tokens ][ ? ]
                              ^
                    if this is row B's space, row A cannot grow
```

You could give row A more room by copying the whole strip to a bigger free
area, but that copy is gigabytes and happens between two tokens that are a
few milliseconds apart. Copying at that rate is impossible, so the engine
never does it. It books the worst case at admission and keeps the booking
until the sequence ends, even if only 0.4% of it is ever used.

Note the slightly killer line: it is not just that row A wastes space. Row A's
booking is *exclusive*: the 99.6% of empty columns belong to row A and no one
else may use them. The machine fills with empty-but-reserved space, and the
scheduler keeps saying no. The fix this lecture sells is giving up on the
continuous strip entirely: cut a sequence into fixed pieces (blocks), scatter
them anywhere, and keep a little table (block table) that says which piece is
where. The pieces need not be next to each other, so nothing has to be copied
when the conversation grows: you just add one more piece.

---

## How can memory be free but unusable?

**Lecture:** [09. Paged attention](09-paged-attention.md)

"Free memory is free memory" is the wrong intuition. A freed strip of memory
comes in the size of the request that freed it, and a next request can only use
a strip that is contiguous and at least as large as its need. A finished
128-token chat leaves a 128-token strip behind; a request that needs 32,768
contiguous tokens cannot live in it. Space that no waiting request can use is
called external fragmentation, and it grows with the variety of admitted
sizes: book everyone at the same maximum and gaps don't arise (instead you pay
reservation waste); book everyone at their own cap and dead strips of odd
sizes accumulate.

Paging removes the "contiguous and large enough" constraint. All blocks are
the same size, so any freed block serves any next request, and a sequence is
reassembled from whatever collection of blocks its block table points to.
Wrong-sized fragments stop existing as a category.

---

## Why can't decode just run the whole answer in one pass, like prefill runs the whole prompt?

**Lecture:** [01. The two phases](01-the-two-phases.md)

Because prefill and decode have different inputs. When you hit enter, the whole
prompt exists at once: every token is on disk or in memory the moment the
request starts, so all of them can be processed in parallel. Decode's input is
the model's own previous output. Token 2 does not exist until token 1 has been
chosen, token 3 until token 2 exists, and so on down the line. Each step is a
prerequisite for the next, so the steps cannot overlap.

Reading is like scanning a page you already hold. Generating is like speaking:
you cannot say word three until word two has left your mouth.

Note what batching does and does not change. Batching parallelizes *across*
sequences (many conversations share one step), never *along* one sequence. No
batching trick makes one conversation's next token computable before the last
one.

---

## But padding is wasted compute. Why not run each sequence on its own and waste nothing?

**Lecture:** [07. Static batching](07-static-batching.md)

Because the expensive waste is not the padding, it is the **memory traffic**.
The weights are 840 MiB, and decode must re-read all of them for every
generated token, for every sequence. Run 32 sequences one at a time and the
weights make 32 separate journeys from memory each step. Batch them and one
journey serves all 32: Lecture 01's batching table shows a batch of 32 moves
the same bytes as a batch of 1 and does 32x the arithmetic on them.

Padding burns arithmetic on positions the mask makes harmless. Decode is
memory-bound, so it has arithmetic to spare: wasting compute on pads costs far
less than repeating weight reads. The attention mask exists to keep padding
from changing the *result*; it is not there to save the *compute*.

---

## How can two blocks hold identical token ids but different histories?

**Lecture:** [10. Prefix caching](10-prefix-caching.md)

The same 16 token ids can appear at different positions in two different
prompts. A token's K/V record is computed from everything the model had seen
before that token: causal attention, Lecture 01. The record stored for
position 2,000 includes 2,000 tokens of context that position 0 never had.
Same tokens, different numbers.

The parent hash is what tells these two cases apart. It fingerprints the
preceding block, so the chain is part of the block's identity: block "abc"
after history X is not the same block as "abc" after history Y. Two stores
with identical ids only merge if their entire histories match.

---

## Why can't a sequence just write into a block it shares?

**Lecture:** [10. Prefix caching](10-prefix-caching.md)

The naive plan: a sequence's next K/V entry lands in the next slot of the
block its block table points to — and if three sequences share that block,
all three would write to the same bytes. K/V are written in place, so the
last writer wins and the other two attend to corrupted memory. The wrong
intuition is that sharing implies write permission.

Copy-on-write fixes it without banning sharing. A block with refcount > 1 is
never written in place: the writer allocates a fresh block, copies the 16
tokens, writes into the copy, decrements the original's refcount, and repoints
its own block table at the copy. Cost: one small block copy, exactly at the
moment a sequence diverges — before that, sharing was free. This is the same
trick an OS uses for `fork()`, and it is what makes parallel sampling and
beam search cheap: diverging alternatives copy one block, pruned alternatives
just decrement.

---

## But kernel launches are asynchronous: can't the CPU just run ahead and hide all this?

**Lecture:** [13. CUDA graphs](13-cuda-graphs.md)

Asynchronous means the CPU submits a kernel and does not wait for it to
finish, not that submitting is free. Each submission still costs the CPU its
own microseconds, 5-10 on this machine, to build and hand over the command.
Small-batch kernels execute in a few microseconds on the chip. When the
per-call CPU cost exceeds the per-kernel chip time, the CPU falls behind: the
chip drains the queue of commands and then sits idle waiting for the next
order.

That is the gap Nsight shows between kernels: the chip is empty because the
CPU is still writing out the next command. CUDA graphs remove the per-call
overhead by binding a whole sequence of kernels into one submission, so the
CPU stops falling behind in the first place.

---

## Why doesn't eager PyTorch fuse these three ops itself?

**Lecture:** [16. Triton basics](16-triton-basics.md)

Because eager PyTorch runs each operation as its own kernel, and nothing tells
it the intermediate result can stay on-chip instead of being written out to
memory. In eager mode, the matmul writes its output to HBM, the broadcast
reads it back, and so on: three kernels, three round-trips between registers
and memory.

Fusion is a *decision* a compiler must make, and Triton lets you make it
explicitly, by writing one kernel instead of three. Lecture 13's
`torch.compile` automates exactly this decision at graph level; Triton is the
manual, in-kernel version. That is why the two techniques compound: one fuses
across operators, the other fuses *within* a single operator.

---

## How can two orderings of floating-point math both be "exact"?

**Lecture:** [17. FlashAttention](17-flash-attention.md)

"Exact" here means no approximation: every score, every weight, every sum is
computed in full, once, in floating point. Nothing is truncated, sampled, or
skipped, the way quantization (Lecture 19) or a genuine approximation would
be. It does not mean bit-for-bit equality with a different arrangement of the
same arithmetic.

FlashAttention computes softmax in row pieces and rescales, which merely
reorders the additions. Floating-point addition is not perfectly associative,
so the last bits of the rounding differ from standard attention, which is why
the test tolerance is around 1e-2 rather than exact match. There is no error
beyond that rounding: the difference is the same kind of noise as running the
same math in a different order by hand.

---

## Why does a running-max softmax recompute the exact same denominator?

**Lecture:** [17. FlashAttention](17-flash-attention.md)

Because rescaling is exactly the arithmetic of "re-normalizing the old sum to
the new max", which is what a one-shot softmax does implicitly in one pass.
Work the same row both ways. Scores `[3, 1, 4, 1, 5, 2]`, tiles of two,
`l = Σ exp(score − m)`:

```
tile 1:  m = 3            l = e⁰ + e⁻²                     = 1.1353
tile 2:  m = 4, move e⁻¹  l = 1.1353·e⁻¹ + e⁰ + e⁻³         = 0.4177 + 1.0498  = 1.4675
tile 3:  m = 5, move e⁻¹  l = 1.4675·e⁻¹ + e⁰ + e⁻³         = 0.5398 + 1.0498  = 1.5896
```

One shot, with the true max 5:

```
l = e⁻² + e⁻⁴ + e⁻¹ + e⁻⁴ + e⁰ + e⁻³  =  0.1353 + 0.0183 + 0.3679 + 0.0183 + 1 + 0.0498  =  1.5896
```

Same number. The move-by-`e^(m_old − m_new)` rule is not an approximation: it
is the algebraic identity `e^(x − m_new) = e^(x − m_old) · e^(m_old − m_new)`
applied to every term at once. The output accumulator rescales by the same
factor, so the quotient `acc / l` is the same weighted average a one-shot pass
computes — which is why the "recompute" framing in the video is really just
"the order of the additions changed".

---

## Why divide by sqrt(d) and not by d?

**Lecture:** [17. FlashAttention](17-flash-attention.md)

Scores are dot products, and a dot product of two `d`-dimensional vectors with
unit-variance entries has variance `d`, not 1: the `d` independent terms add
their variances instead of averaging. So the standard deviation of a score is
`√d`, which at `d = 64` is **8** — scores scatter "±8", not the "±1" people
guess — and the typical gap between the best and worst key is ~2 sd ≈ 16:

```
e¹⁶ ≈ 8.9 million
```

`exp` turns that gap into a near-total preference for one token. On the toy
scores `[0, 6, 2, 2]` (a 4-dim dot product) the top token takes 96.2% of the
softmax weight:

```
e⁶ / (e⁶ + e⁰ + 2e²)  =  403.4 / 419.2  =  96.2%
```

A spiked softmax is nearly an argmax. Attention "listens" to one key, and the
gradient through it collapses: a softmax row's Jacobian is `p(δ − p)` (the
identity minus the outer product of `p`, scaled by `p`), which is ~0 wherever
one entry is ~1, so nothing learns. Dividing by `√d` first restores sd 1, the
gaps shrink to ~2, `e²` is a 7:1 ratio, and the same toy stays soft — scaling
`[0, 6, 2, 2]` by 1/√4 gives `[0, 3, 1, 1]`, top weight 76%. The scalar
controls the *spread* of the scores, not the scale of any one value.

One subtlety: the standard shorthand "restores the variance to exactly one"
means the variance of each *score* (the dot product as a whole), not of each
component — the components keep their variance, the sum is what lands on 1.

---

## The GQA video says the KV cache dwarfs the weights; the book says weights are 92% of decode traffic

**Lecture:** [01. The two phases](01-the-two-phases.md) and [05. The KV cache](05-kv-cache.md)

Both are right, on different sides of the same line — the ~8k-context
crossover. The book's 92.3% is the *per-token* decode ledger at short context
(215,040 MiB of 232,960 MiB total is the weights, Lecture 01). The video's
"the KV cache dwarfs the weights" is the *stored-size* claim at long context
(Lecture 05: at 32k the cache is 3.5 GiB, 4.3× the model).

The practical question is where halving the KV heads (16 → 8, grouped-query
attention) actually moves the needle. Per token, K and V cost 112 KiB across
28 layers (Lecture 05's formula); weights are 880.8 MB:

```
context 512:  cache 56 MiB → 28 MiB   total 910.2 → 908.8 MB   ≈  3% cut
context 8k:   cache 896 MiB → 448 MiB total 1776.8 → 1328.8 MB  ≈  26% cut
```

At 512 tokens the halving is rounding error; at 8k it is a quarter of the
traffic. Serving hurts at long context *and* large batch, where the cache is
both big and re-read every step — so the video and the book describe the two
ends of one line, and GQA's payoff lives at the video's end.

---

## Wait: "block" here is a thread block, not the KV block from Lecture 09?

**Lecture:** [18. Paged attention kernel](18-paged-attention-kernel.md)

Both words are in play, and they mean different things.

The **KV block** is a fixed-size chunk of cached data, 16 tokens, from Lecture
09: an allocation unit. You allocate it, fill it, free it.

The **thread block** is a group of threads that run together on one SM (one
of the chip's work groups, each with its own fast private memory): an
execution unit. Splitting the context across thread blocks means different
groups of threads each handle a slice of the sequence.

Same word, unrelated meanings. The sentence around it tells you which is
meant: blocks you allocate and free are KV blocks, blocks of threads that
execute are thread blocks.

---

## Why can't the API server just be a thread in the same process?

**Lecture:** [24. Serving](24-serving.md)

Python threads share the GIL, so they do not run in parallel: the HTTP thread
and the engine thread would take turns inside the same interpreter. The HTTP
work would still execute *between* engine steps, and every request on the wire
would wait for the engine's chunk boundary.

A separate process gets its own interpreter and its own event loop, which run
truly at the same time as the engine. The engine's only contact with the
outside world is the two queues it already has: requests in, responses out.
That is the whole point of the decoupled design: neither side waits on the
other's clock.

---

## Arrivals average 20 req/s and the server keeps up on average. Why does a queue ever form?

**Lecture:** [25. Load testing](25-load-testing.md)

Because "on average" describes no particular moment. A Poisson process with a
mean gap of 50 ms still produces bursts: several arrivals within a few
milliseconds, then a lull. During the burst, arrivals can outpace the server
for a few hundred milliseconds, and a queue forms.

The queue only drains when a quiet gap arrives, and only if the long-run
arrival rate stays below capacity. Past the knee, arrival rate sits at or
above capacity, so the bursts never fully drain: the queue grows without
bound. Average traffic keeps up; momentary traffic decides the latency tail.

---

## If round-robin throws away the cache, why would anyone use it?

**Lecture:** [27. Routing and disaggregation](27-routing-and-disaggregation.md)

Round-robin is stateless. It needs no coordination between replicas, no
telemetry, no shared bookkeeping, and it balances load exactly: request N
goes to replica N mod R, always. Those properties matter.

On traffic with little repetition, the cache-aware alternative pays
bookkeeping costs (hash computation, coordination, redirects) while the cache
itself saves almost nothing, because each prompt is new. Cache-aware routing
wins in exactly the regime where prefixes repeat. Which one serves you is a
tuning decision, and the lecture's point is that the cache-aware costs only
earn their keep when they touch a real hit rate.

---

## How can the GPU report 95% busy and be nearly idle at the same time?

**Lecture:** [28. Autoscaling and cost](28-autoscaling-and-cost.md)

`nvidia-smi` counts "busy" as time with at least one kernel running. Decode is
memory-bound: the kernel runs the whole time, so the GPU is 100% "busy" while
the compute units wait on memory to deliver the next bytes. Busyness measures
whether anything is executing, not whether it is making progress.

The two metrics answer different questions. Utilization says: is the chip ever
completely empty? Queue depth says: is work waiting for a free slot? Under
memory-bound decode you can have the first high and the second low
simultaneously, and the queue is the one that tells you whether to scale.

---

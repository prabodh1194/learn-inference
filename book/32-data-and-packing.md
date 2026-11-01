# 32. Data pipelines, sequence packing, and loss masking

**Demo:** `book/code/packing_waste.py` · **Build:** `train/packing.py` · **Test:** `tests/test_32_packing.py`
**Moves:** eliminates 58% padded FLOPs ($2.28\times$ throughput); fixes the gradient-accumulation scaling bug
**Prereq:** [31. Activations and checkpointing](31-activation-memory.md), [07. Static batching](07-static-batching.md)

---

## The problem

In Lecture 07 (Static batching), padding a batch of variable-length prompts to
the longest sequence wasted 64% of prefill slots. Back then, decode was
memory-bound, so extra padded arithmetic rode along almost for free on spare
compute units.

**Training has no memory-bound decode phase.** Every training step is a large,
compute-bound forward and backward pass ($6ND$ FLOPs, Lecture 30) sitting far to
the right of the roofline ridge point. Every `[PAD]` token you feed into a
training batch costs the **exact same $6N$ FLOPs and $38hL$ bytes of activation
memory as a real token** — and then its gradient is multiplied by zero at the
loss function and thrown in the trash.

Worse: when you combine variable-length sequences with gradient accumulation
(Lecture 31), PyTorch's default `CrossEntropyLoss(reduction="mean")` introduces
a silent mathematical bug that weights short sequences up to **$10\times$ more
heavily per token** than long ones.

---

## See it

```bash
uv run python book/code/packing_waste.py
```

```
64 documents (exponential lengths, mean ~428 tokens, cap 2048)
=========================================================================
  padded batches (batch=8)   : 31,307 useful / 74,704 allocated  (58.1% wasted FLOPs)
  packed into 2048-tok bins  : 31,307 useful / 32,768 allocated  ( 4.5% slack across 16 bins)
  effective throughput gain  : 2.28x fewer padded token-slots

The gradient-accumulation averaging trap (2 micro-batches)
=========================================================================
  micro-batch 1: 100 valid tokens @ loss 2.0/token  (total = 200.0)
  micro-batch 2:  10 valid tokens @ loss 5.0/token  (total =  50.0)
  naive (L1 + L2) / 2          : 3.5000  (weights each MB2 token 10x heavier!)
  true  sum(loss) / sum(tokens): 2.2727
```

Look at both numbers:

1. **58.1% of your GPU bill vanishes into padding** on a realistic exponential
   length distribution ($31{,}307$ real tokens out of $74{,}704$ allocated slots).
   Packing the same 64 documents into 16 bins of length $2{,}048$ cuts waste to
   **4.5%** — a **$2.28\times$ speedup** ($74{,}704 / 32{,}768 = 2.28\times$)
   without touching a single kernel.
2. **Naive micro-batch averaging gives $3.5000$ instead of the true per-token
   loss $2.2727$** — a $54\%$ distortion of the objective function!

---

## The idea

### 1. Sequence packing and the cross-document leakage trap

Instead of padding each short document to $S = 2048$, **concatenate** multiple
documents end-to-end into a single sequence of length $S = 2048$, separated by
`<|endoftext|>` (`EOS`):

```
PADDED (3 rows × 8 slots = 24 token-slots, 11 wasted):
  row 0: [docA_0, docA_1, docA_2, EOS,    PAD,    PAD,    PAD,    PAD   ]
  row 1: [docB_0, docB_1, EOS,    PAD,    PAD,    PAD,    PAD,    PAD   ]
  row 2: [docC_0, docC_1, docC_2, docC_3, docC_4, EOS,    PAD,    PAD   ]

PACKED (2 rows × 8 slots = 16 token-slots, 1 wasted):
  bin 0: [docA_0, docA_1, docA_2, EOS,    docB_0, docB_1, EOS,    PAD   ]
  bin 1: [docC_0, docC_1, docC_2, docC_3, docC_4, EOS,    ...           ]
```

Now look closely at `bin 0`. If you hand `bin 0` to standard causal attention,
token `docB_0` (at index 4) attends to `docA_0, docA_1, docA_2, EOS` (indices
0–3), and its RoPE position embedding says it is at position `4` instead of
position `0`!

Two things must be reset at every document boundary inside a packed bin:

1. **`position_ids` must reset to `0` at the start of each packed document** so
   RoPE (Rotary Position Embedding) sees relative positions inside each
   document, not the offset inside the bin:
   ```
   tokens       : [docA_0, docA_1, docA_2, EOS, docB_0, docB_1, EOS]
   position_ids : [     0,      1,      2,   3,      0,      1,   2]
   ```
2. **The attention mask must be block-diagonal causal**, forbidding any token in
   Document B from attending to Document A:
   ```
             docA (len 4)      docB (len 3)
          ┌─────────────────┬─────────────┐
          │ 1   0   0   0   │ 0   0   0   │
   docA   │ 1   1   0   0   │ 0   0   0   │
          │ 1   1   1   0   │ 0   0   0   │
          │ 1   1   1   1   │ 0   0   0   │
          ├─────────────────┼─────────────┤
          │ 0   0   0   0   │ 1   0   0   │
   docB   │ 0   0   0   0   │ 1   1   0   │
          │ 0   0   0   0   │ 1   1   1   │
          └─────────────────┴─────────────┘
   ```
3. **The first token of Document B must not be scored as the "next token" after
   Document A's `EOS`!** Set `labels = -100` at the boundary transition (or
   shift labels *within* each document before packing).

??? question "Wait — if FlashAttention never materializes the $S \times S$ mask, how does it enforce a block-diagonal mask across packed documents?"
    Through `cu_seqlens` (cumulative sequence lengths) in
    `flash_attn_varlen_func`. Instead of passing a 2-D `(S, S)` mask tensor,
    you flatten all packed documents into a 1-D stream of tokens and pass a 1-D
    integer array of boundary offsets, e.g. `cu_seqlens = [0, 120, 420, 500]`
    for documents of lengths `120`, `300`, and `80`. Each thread block in the
    FlashAttention kernel reads `cu_seqlens[b]` and `cu_seqlens[b+1]` and loops
    **only** over key/value tiles inside its own document!
    [Full answer](qa.md#how-does-flashattention-enforce-block-diagonal-document-masks-without-an-sxs-mask)

That `cu_seqlens` trick gives a **second** speedup beyond eliminating padding:
it skips the off-diagonal blocks entirely! If a $4{,}096$-token bin holds 8
documents of length $s_i = 512$, standard causal attention over the whole
$4{,}096$-token bin computes proportional to:

$$S^2 = 4096^2 = 16{,}777{,}216 \text{ score pairs}$$

while `flash_attn_varlen_func` with `cu_seqlens = [0, 512, 1024, ..., 4096]`
computes only the 8 diagonal blocks:

$$\sum_{i=1}^8 s_i^2 = 8 \times 512^2 = 8 \times 262{,}144 = 2{,}097{,}152 \text{ score pairs}$$

Ratio:

$$\frac{16{,}777{,}216}{2{,}097{,}152} = 8.0\times \text{ less attention compute}$$

---

### 2. Deriving the gradient-accumulation token-averaging bug

Suppose you accumulate gradients over $M = 2$ micro-batches (or across $M = 2$
data-parallel GPUs). Let $\ell_{m, t}$ be the cross-entropy loss on valid
(non-masked) token $t$ in micro-batch $m$, and let $N_m$ be the number of valid
tokens in micro-batch $m$.

The **true mean loss per valid token** over the combined batch is the sum of all
token losses divided by the total number of valid tokens:

$$\mathcal{L}_{\text{true}} = \frac{\sum_{m=1}^M \sum_{t=1}^{N_m} \ell_{m,t}}{\sum_{m=1}^M N_m}$$

Now see what happens when you write the textbook gradient-accumulation loop
using PyTorch's default `F.cross_entropy(logits, labels, ignore_index=-100)`
(which defaults to `reduction="mean"` inside each micro-batch):

$$\mathcal{L}_m = \frac{1}{N_m} \sum_{t=1}^{N_m} \ell_{m,t}$$

and average those $M$ numbers via `loss = loss_m / M`:

$$\mathcal{L}_{\text{naive}} = \frac{1}{M} \sum_{m=1}^M \mathcal{L}_m = \sum_{m=1}^M \sum_{t=1}^{N_m} \left(\frac{1}{M \cdot N_m}\right) \ell_{m,t}$$

Compare the weight placed on each individual token $\ell_{m,t}$ in the two
equations:

- In $\mathcal{L}_{\text{true}}$, every valid token in the global batch has the
  **same weight**:
  $$w_{\text{true}} = \frac{1}{N_1 + N_2 + \dots + N_M}$$
- In $\mathcal{L}_{\text{naive}}$, a token in micro-batch $m$ has weight
  $$w_{\text{naive}}(m) = \frac{1}{M \cdot N_m}$$
  which is **inversely proportional to $N_m$**!

Plug in the demo's numbers:
- Micro-batch 1 has $N_1 = 100$ valid tokens, each with loss $\ell_{1,t} = 2.0$,
  so $\sum_t \ell_{1,t} = 200.0$ and $\mathcal{L}_1 = 200.0 / 100 = 2.0$.
- Micro-batch 2 has $N_2 = 10$ valid tokens, each with loss $\ell_{2,t} = 5.0$,
  so $\sum_t \ell_{2,t} = 50.0$ and $\mathcal{L}_2 = 50.0 / 10 = 5.0$.

Compute both:

```
naive average:  (L_1 + L_2) / 2               =  (2.0 + 5.0) / 2        =  3.5000
true average:   (sum_1 + sum_2) / (N_1 + N_2) =  (200.0 + 50.0) / 110   =  2.2727
```

In the naive loop, each token in micro-batch 2 gets weight
$\frac{1}{2 \times 10} = \frac{1}{20}$, while each token in micro-batch 1 gets
weight $\frac{1}{2 \times 100} = \frac{1}{200}$ — a **$10\times$ bias** toward
whichever micro-batch happened to have more padding or longer masked prompts!

??? question "When does this bug actually bite in practice?"
    Every time `N_m` varies across micro-batches or across data-parallel ranks —
    which is **always** in SFT and RL post-training (Lectures 36–39), where
    prompts are masked with `-100` and response lengths vary from 10 tokens to
    4,000 tokens. It was a famous real-world bug fixed in HuggingFace
    `transformers` (`Trainer` + `accelerate`) in late 2024: the fix is to compute
    `F.cross_entropy(..., reduction="sum")` on each micro-batch and divide by
    the total count of non-`-100` tokens across the entire global step.
    [Full answer](qa.md#why-does-averaging-micro-batch-losses-bias-gradients-toward-short-completions)

---

### 3. Pretraining data curation & MinHash LSH deduplication

Before a single token reaches the sequence packer, a pretraining pipeline
(FineWeb-Edu, DCLM, Stanford CS336 Assignment 4, Qwen3's $36\text{T}$-token
corpus) transforms petabytes of raw Common Crawl HTML into clean documents
through four stages:

1. **Text extraction & heuristic filtering:** Strip HTML boilerplates (`trafilatura`
   / `resiliparse`), detect language (`fastText`), and discard documents with
   abnormal symbol-to-word ratios or repeated lines.
2. **Fuzzy deduplication via MinHash LSH:** Web crawls contain up to $30\%$–$50\%$
   near-duplicate boilerplate. Comparing all $\binom{N}{2}$ document pairs for
   $N = 10^9$ documents would require $5 \times 10^{17}$ comparisons.
   **MinHash Locality-Sensitive Hashing (LSH)** reduces this to $O(N)$ by hashing
   each document's set of $n$-grams into $K = b \cdot r$ MinHash signatures split
   into $b$ **bands** of $r$ **rows** each.

#### Full Derivation of the MinHash LSH S-Curve

Let two documents $A$ and $B$ have $n$-gram Jaccard similarity:

$$s = J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

By a fundamental property of random permutations, a single MinHash function $h_i$
produces the same minimum hash for $A$ and $B$ with probability exactly equal to
their Jaccard similarity:

$$\Pr[h_i(A) = h_i(B)] = s$$

Now partition $K = b \cdot r$ independent MinHash values into $b$ bands of $r$
rows each. Two documents are flagged as a candidate duplicate pair if and only if
they match on **all $r$ rows in at least one of the $b$ bands**:

1. **Probability that all $r$ rows match in one specific band:** Because the $r$
   hash functions are independent,
   $$\Pr[\text{band } j \text{ matches}] = \underbrace{s \cdot s \cdots s}_{r \text{ times}} = s^r$$
2. **Probability that band $j$ does *not* match:**
   $$\Pr[\text{band } j \text{ differs}] = 1 - s^r$$
3. **Probability that *none* of the $b$ independent bands match:**
   $$\Pr[\text{no band matches}] = (1 - s^r)^b$$
4. **Probability that *at least one* band matches (candidate collision):**
   $$P_{\text{collision}}(s) = 1 - (1 - s^r)^b$$

Plug in standard FineWeb parameters ($K = 100$ hashes split into $b = 20$ bands
of $r = 5$ rows, threshold $t \approx (1/b)^{1/r} = (1/20)^{1/5} \approx 0.549$):

- **Near-duplicates ($s = 0.80$):**
  $$s^5 = 0.80^5 = 0.32768 \implies P_{\text{collision}}(0.80) = 1 - (1 - 0.32768)^{20} = 1 - 0.67232^{20} = 1 - 0.00035 = 99.96\%$$
- **Unrelated documents ($s = 0.30$):**
  $$s^5 = 0.30^5 = 0.00243 \implies P_{\text{collision}}(0.30) = 1 - (1 - 0.00243)^{20} = 1 - 0.99757^{20} = 4.75\%$$

3. **Model-based quality filtering & synthetic data:** Modern corpora (FineWeb-Edu,
   DCLM, Qwen3) score every deduplicated document with a lightweight classifier
   distilled from LLM educational-value annotations, then augment natural web
   text with trillions of synthetic math, code, and textbook tokens.
4. **Domain & instance-level mixing:** Rather than sampling domains in proportion
   to raw web volume, small proxy models optimize domain weights (DoReMi, RegMix,
   or Qwen3's instance-level mixture optimization) to up-weight STEM and code.

---

## The code

```python
def pack_sequences(lengths: list[int], max_seq_len: int) -> list[list[int]]:
    bins: list[list[int]] = []
    for length in lengths:
        for b in bins:
            if sum(b) + length <= max_seq_len:
                b.append(length)
                break
        else:
            bins.append([length])
    return bins


def cu_seqlens_from_pack(doc_lengths: list[int]) -> list[int]:
    out = [0]
    for length in doc_lengths:
        out.append(out[-1] + length)
    return out


def token_weighted_loss(sum_losses: list[float], token_counts: list[int]) -> float:
    return sum(sum_losses) / sum(token_counts)
```

---

## Build it

1. Implement `pack_sequences`, `cu_seqlens_from_pack`, and `token_weighted_loss`
   in `train/packing.py`.
2. Run the tests:

```bash
uv run pytest tests/test_32_packing.py -v
```

3. Check that `cu_seqlens_from_pack([120, 300, 80])` returns `[0, 120, 420, 500]`
   (length $K + 1$ for $K$ documents).

---

## What you should see

- **Packing turns 58% padding waste into 4.5% bin slack**, cutting the number of
  token-slots processed by $2.28\times$.
- **Token-weighted loss is invariant to how documents are split across
  micro-batches**, whereas naive mean-of-means changes the loss and gradient
  whenever valid token counts differ.

---

## Go deeper

- **[FlashAttention-2 `flash_attn_varlen_func`](https://github.com/Dao-AILab/flash-attention)** — how variable-length packed sequences execute without padding or cross-document attention.
- **[Unsloth / HuggingFace Gradient Accumulation Bug Write-up](https://huggingface.co/blog/gradient_accumulation)** — how `reduction="mean"` across micro-batches silently degraded SFT runs across the industry.

---

## Check yourself

1. Why is padding waste much more expensive in training than in single-batch
   decode? (Hint: where does each sit on Lecture 02's roofline?)
2. What two things go wrong if you concatenate three documents into one
   $4{,}096$-token sequence and feed it to a standard causal transformer without
   changing `position_ids` or `cu_seqlens`?
3. Micro-batch 1 has 500 valid tokens with mean loss $1.0$; micro-batch 2 has
   50 valid tokens with mean loss $4.0$. Compute both the naive mean and the
   true per-token loss.

---

## Next

**[33. Data parallelism, ZeRO-1/2/3, and FSDP](33-zeRO-and-fsdp.md)**:
one GPU is full. Now scale across 8 GPUs — and stop holding 8 identical copies
of that 8.88 GiB optimizer state.

```bash
uv run python book/code/zero_memory.py
```

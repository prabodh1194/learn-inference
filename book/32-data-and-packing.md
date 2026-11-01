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
training batch costs the **exact same $6N$ FLOPs and $38hL$ bytes of saved
tensors as a real token**, and then the loss function multiplies its gradient
by zero and throws the work away.

Worse still, when you pair variable-length sequences with gradient accumulation
(Lecture 31), PyTorch's default `CrossEntropyLoss(reduction="mean")` introduces
a silent math bug that weights short sequences up to **$10\times$ more heavily
per token** than long ones.

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

Look at both numbers in that output. On a realistic exponential length
distribution, **58.1% of your GPU bill vanishes into padding**, with only
$31{,}307$ real tokens out of $74{,}704$ allocated slots. When we pack those
same 64 documents into 16 bins of length $2{,}048$, waste drops to **4.5%**.
That gives us a **$2.28\times$ speedup** ($74{,}704 / 32{,}768 = 2.28\times$)
without touching a single kernel. At the same time, naive micro-batch averaging
reports a loss of **$3.5000$** when the true per-token loss is **$2.2727$**, a
$54\%$ error in the objective we are trying to minimize.

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
token `docB_0` at index 4 attends back to `docA_0, docA_1, docA_2, EOS` at
indices 0 to 3, and its RoPE position index says `4` when `docB_0` is really the
first token (`0`) of Document B.

To pack documents without leaking text from Document A into Document B, we have
to reset three things at every document boundary inside a bin.

First, `position_ids` must reset to `0` at the start of each packed document so
RoPE sees each token's offset inside its own document, not its offset inside the
bin:

```
tokens       : [docA_0, docA_1, docA_2, EOS, docB_0, docB_1, EOS]
position_ids : [     0,      1,      2,   3,      0,      1,   2]
```

Second, the attention mask must be **block-diagonal causal**, blocking any token
in Document B from reading keys and values from Document A:

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

Third, the first token of Document B must not be scored as the next token after
Document A's `EOS`. We either mask that boundary target with `labels = -100` or
shift inputs and targets inside each document before packing them together.

??? question "Wait, if FlashAttention never builds the $S \times S$ mask, how does it enforce a block-diagonal mask across packed documents?"
    Through `cu_seqlens` (cumulative sequence lengths) in
    `flash_attn_varlen_func`. Instead of passing a 2-D `(S, S)` mask tensor,
    you flatten all packed documents into a 1-D stream of tokens and pass a 1-D
    integer array of boundary offsets, such as `cu_seqlens = [0, 120, 420, 500]`
    for documents of lengths `120`, `300`, and `80`. Each thread block in the
    FlashAttention kernel reads `cu_seqlens[b]` and `cu_seqlens[b+1]` and loops
    **only** over key/value tiles inside its own document.
    [Full answer](qa.md#how-does-flashattention-enforce-block-diagonal-document-masks-without-an-sxs-mask)

That `cu_seqlens` trick gives us a **second** speedup beyond cutting padding:
it skips the off-diagonal blocks altogether. If a $4{,}096$-token bin holds 8
documents of length $s_i = 512$, standard causal attention over the whole
$4{,}096$-token bin computes score pairs proportional to $S^2$.

$$S^2 = 4096^2 = 16{,}777{,}216 \text{ score pairs}$$

When we call `flash_attn_varlen_func` with `cu_seqlens = [0, 512, 1024, ..., 4096]`,
the kernel computes only the 8 diagonal blocks.

$$\sum_{i=1}^8 s_i^2 = 8 \times 512^2 = 8 \times 262{,}144 = 2{,}097{,}152 \text{ score pairs}$$

Take the ratio of those two counts.

$$\frac{16{,}777{,}216}{2{,}097{,}152} = 8.0\times \text{ less attention compute}$$

---

### 2. Deriving the gradient-accumulation token-averaging bug

Suppose we accumulate gradients over $M = 2$ micro-batches, or over $M = 2$
data-parallel GPUs. Let $\ell_{m, t}$ be the cross-entropy loss on valid
(non-masked) token $t$ in micro-batch $m$, and let $N_m$ be the number of valid
tokens in micro-batch $m$.

The **true mean loss per valid token** over the combined batch adds up all token
losses and divides by the total number of valid tokens.

$$\mathcal{L}_{\text{true}} = \frac{\sum_{m=1}^M \sum_{t=1}^{N_m} \ell_{m,t}}{\sum_{m=1}^M N_m}$$

Now see what happens when we write the textbook loop using PyTorch's default
`F.cross_entropy(logits, labels, ignore_index=-100)`, which defaults to
`reduction="mean"` inside each micro-batch.

$$\mathcal{L}_m = \frac{1}{N_m} \sum_{t=1}^{N_m} \ell_{m,t}$$

If we average those $M$ numbers via `loss = loss_m / M`, we get a different sum.

$$\mathcal{L}_{\text{naive}} = \frac{1}{M} \sum_{m=1}^M \mathcal{L}_m = \sum_{m=1}^M \sum_{t=1}^{N_m} \left(\frac{1}{M \cdot N_m}\right) \ell_{m,t}$$

Compare the weight placed on each individual token $\ell_{m,t}$ in the two
equations. In $\mathcal{L}_{\text{true}}$, every valid token in the global batch
gets the **same weight**.

$$w_{\text{true}} = \frac{1}{N_1 + N_2 + \dots + N_M}$$

In $\mathcal{L}_{\text{naive}}$, a token in micro-batch $m$ gets weight
$w_{\text{naive}}(m) = \frac{1}{M \cdot N_m}$, which scales **inversely with
$N_m$**.

Plug in the demo's numbers. Micro-batch 1 has $N_1 = 100$ valid tokens, each
with loss $\ell_{1,t} = 2.0$, so $\sum_t \ell_{1,t} = 200.0$ and
$\mathcal{L}_1 = 200.0 / 100 = 2.0$. Micro-batch 2 has $N_2 = 10$ valid tokens,
each with loss $\ell_{2,t} = 5.0$, so $\sum_t \ell_{2,t} = 50.0$ and
$\mathcal{L}_2 = 50.0 / 10 = 5.0$.

Compute both:

```
naive average:  (L_1 + L_2) / 2               =  (2.0 + 5.0) / 2        =  3.5000
true average:   (sum_1 + sum_2) / (N_1 + N_2) =  (200.0 + 50.0) / 110   =  2.2727
```

In the naive loop, each token in micro-batch 2 gets weight
$\frac{1}{2 \times 10} = \frac{1}{20}$, while each token in micro-batch 1 gets
weight $\frac{1}{2 \times 100} = \frac{1}{200}$. That gives a **$10\times$ bias**
toward whichever micro-batch happened to have more padding or longer masked
prompts.

??? question "When does this bug actually bite in practice?"
    Every time `N_m` varies from one micro-batch or data-parallel rank to the
    next. That happens on every step in SFT and RL post-training (Lectures 36 to
    39), where prompts are masked with `-100` and replies range from 10 tokens to
    4,000 tokens. It was a famous real-world bug fixed in HuggingFace
    `transformers` (`Trainer` + `accelerate`) in late 2024: the fix computes
    `F.cross_entropy(..., reduction="sum")` on each micro-batch and divides by
    the total count of non-`-100` tokens in the global step.
    [Full answer](qa.md#why-does-averaging-micro-batch-losses-bias-gradients-toward-short-completions)

---

### 3. Pretraining data curation and MinHash LSH deduplication

Before a single token reaches the sequence packer, we have to turn raw web
crawls into clean training documents. Modern pipelines like FineWeb-Edu, DCLM,
and Qwen3's $36\text{T}$-token corpus start by stripping HTML tags and page
chrome with tools like `trafilatura` or `resiliparse`, tagging language with
`fastText`, and dropping pages with garbled symbol-to-word ratios.

Even after stripping HTML, raw web crawls still cannot go straight into training
because $30\%$ to $50\%$ of the pages are near-duplicates (syndicated articles,
mirrors, and boilerplate). Training on those repeated pages biases the model and
burns trillions of FLOPs re-learning the same text.

To drop near-duplicates, we first need a way to measure how much text two
documents share. We measure text overlap with Jaccard similarity,
$s = |A \cap B| / |A \cup B|$. Computing Jaccard requires representing each
document as a set, so we slice each document into a set of word $n$-grams
(typically 5-grams).

Now each document is a set of $n$-grams, but exact Jaccard over $N = 10^9$ web
pages hits an $O(N^2)$ scaling wall of $\binom{10^9}{2} \approx 5 \times 10^{17}$
pairwise set checks. So we use a two-step method: **MinHash** to compress each
document's $n$-gram set into $K$ integers, followed by **Locality-Sensitive
Hashing (LSH)** to bucket candidate matches in $O(N)$ time.

Why does MinHash preserve Jaccard similarity? Pick a random permutation $h_i$ of
all possible $n$-grams and record the minimum hash value $h_i(A)$ in document
$A$. The minimum hash of $A \cup B$ must lie in $A \cap B$ whenever $A$ and $B$
share that minimum element, which happens with probability equal to their
Jaccard overlap.

$$\Pr[h_i(A) = h_i(B)] = \frac{|A \cap B|}{|A \cup B|} = s$$

Comparing all $K$ hashes between every pair of documents would still take
$O(N^2)$ time. LSH avoids the pairwise scan by splitting the $K = b \cdot r$
independent MinHash values into $b$ **bands** of $r$ **rows** each, and hashing
each band into buckets. Two documents become a candidate duplicate pair if and
only if they land in the same bucket in **at least one of the $b$ bands**, which
means all $r$ rows in that band match.

We can derive the collision probability step by step from $s$. First, because the
$r$ hash functions inside one band are independent, the probability that all $r$
rows match in band $j$ is the product of $r$ matches.

$$\Pr[\text{band } j \text{ matches}] = \underbrace{s \cdot s \cdots s}_{r \text{ times}} = s^r$$

Second, the probability that band $j$ does *not* match on all $r$ rows is one
minus $s^r$.

$$\Pr[\text{band } j \text{ differs}] = 1 - s^r$$

Third, because all $b$ bands use independent hashes, the probability that *none*
of the $b$ bands match is the product over all $b$ bands.

$$\Pr[\text{no band matches}] = (1 - s^r)^b$$

Fourth, the probability that *at least one* band matches and flags the pair is
one minus the probability that none match.

$$P_{\text{collision}}(s) = 1 - (1 - s^r)^b$$

That formula traces a sharp S-curve with inflection threshold
$t \approx (1/b)^{1/r}$. In FineWeb, $K = 100$ hashes are split into $b = 20$
bands of $r = 5$ rows, putting the threshold at $t \approx (1/20)^{1/5} \approx 0.549$.
For two near-duplicate pages with $s = 0.80$, a single band matches with
probability $0.80^5 = 0.32768$, so the collision probability over 20 bands is
99.96%.

$$P_{\text{collision}}(0.80) = 1 - (1 - 0.32768)^{20} = 1 - 0.67232^{20} = 1 - 0.00035 = 99.96\%$$

For two unrelated pages with $s = 0.30$, a single band matches with probability
$0.30^5 = 0.00243$, so the false-positive rate over 20 bands is only 4.75%.

$$P_{\text{collision}}(0.30) = 1 - (1 - 0.00243)^{20} = 1 - 0.99757^{20} = 4.75\%$$

Once LSH deduplication finishes, modern pipelines run two final stages. A
lightweight quality classifier (distilled from LLM ratings in FineWeb-Edu, DCLM,
and Qwen3) scores every surviving page for educational value and blends in
synthetic math, code, and textbook tokens. Finally, small proxy models tune the
domain sampling weights (via DoReMi, RegMix, or Qwen3's instance-level mixture
weights) so high-signal code and STEM pages are sampled far more often than raw
web volume would give them.

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

Packing turns **58% padding waste into 4.5% bin slack**, cutting the number of
token-slots processed by $2.28\times$. At the same time, **token-weighted loss**
gives the exact same result no matter how documents are split across
micro-batches, whereas naive mean-of-means shifts the loss and gradient whenever
valid token counts differ.

---

## Go deeper

- [FlashAttention-2 `flash_attn_varlen_func`](https://github.com/Dao-AILab/flash-attention), how variable-length packed sequences run without padding or cross-document attention.
- [Fixing Gradient Accumulation in Transformers](https://huggingface.co/blog/gradient_accumulation), how `reduction="mean"` across micro-batches silently hurt SFT runs across the field.
- [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557) (Penedo et al., 2024), MinHash LSH deduplication and FineWeb-Edu classifier filtering.

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
one GPU is full. Now scale across 8 GPUs and stop holding 8 identical copies of
that 8.88 GiB optimizer state.

```bash
uv run python book/code/zero_memory.py
```

# 01. The two phases

**Demo:** `book/code/two_phases.py` · **Moves:** nothing yet: this is the map
**Prereq:** [00. Introduction](00-intro.md)

---

## The problem

Generating a response looks like one operation. It is two, and they behave so
differently that production systems eventually run them on separate machines.

**Prefill** processes your prompt. All of it, at once, in a single forward pass.
The model computes keys and values for every prompt token in parallel and
produces the first output token. This determines **time to first token (TTFT)**:
how long before the cursor starts moving.

**Decode** generates the rest. One token per forward pass, each depending on the
one before, so there's nothing to parallelize across. Five hundred output tokens
means five hundred sequential passes. This determines **tokens per second (TPS)**:
how fast the text streams.

Same weights, same kernels. Completely different performance characteristics.

---

## See it

```bash
uv run python book/code/two_phases.py
```

Pure arithmetic, no GPU, no model download. Look for three things.

**First**, for a 512-token prompt and 256 tokens out:

```
                 compute     memory read    ops:byte
prefill         451.0 GF          840 MiB      512.00
decode          225.5 GF       232960 MiB        0.92
```

Decode does *half* the compute of prefill and moves **277× more memory**. Both
claims are two divisions:

```
compute:   2 × 440.4M params × 512 prompt tokens  = 451.0 GF    (prefill)
           2 × 440.4M params × 256 output tokens  = 225.5 GF    (decode)
           ratio: 225.5 / 451.0  =  0.5×

memory:    232,960 / 840  =  277.3×
```

That 232,960 is worth deriving rather than accepting, because it's the whole
argument of this lecture in one number. The demo prints the breakdown:

```
  weights, re-read once per token      840 MiB x 256 =    215,040 MiB
  KV cache, re-read and growing                          17,920 MiB
                                  total      232,960 MiB
```

**The first line is 92% of it, and it's one multiplication:**

```
215,040 / 232,960  =  92.3%
```

840 MiB of weights, re-read to produce *each* of the 256 tokens. Prefill reads
the same 840 MiB *once*, for all 512 prompt tokens.

The second line is the KV cache, re-read every step and growing as you go. The
three factors, one at a time (112 KiB = 2 × 28 × 8 × 128 × 2 bytes per token,
derived in Lecture 05):

```
context runs 512 → 768 tokens, so on average      (512 + 768) / 2 = 640 tokens
per step, that's 640 × 112 KiB = 71,680 KiB      = 70 MiB read back
over 256 decode steps: 70 MiB × 256              = 17,920 MiB
```

Ignorable here at 7.7% (17,920 / 232,960); it takes over past ~8k context,
which is a different problem.

**The shape to remember is `weights × tokens generated`.** Not the number.

??? note "Sanity-check it against the hardware"
    One more step, and it becomes a prediction you can falsify:

    ```
    232,960 MiB = 227.5 GiB = 2.44 × 10¹¹ bytes
    2.44 × 10¹¹ B / 936.2 GB/s  =  0.26 s            (bandwidth floor)
    256 tokens / 0.26 s         =  ~985 tok/s         (single-stream ceiling)
    ```

    ~980 tok/s is the *ceiling* for a single stream on a 3090, set purely by
    bandwidth. Real engines land well under it, and if you ever measure above
    it, your measurement is wrong.

    More in the [Q&A](qa.md#where-does-232960-mib-come-from).

**Second**, the per-token table. A generated token costs hundreds to thousands of
times the memory traffic of a prompt token, and the ratio **widens** as prompts
get longer, from 34× at a 32-token prompt to 2596× at 2048:

```
ratio  =  (decode bytes per token) / (prefill bytes per token)

32-token prompt,  1024 out:   900 MiB ÷ 26.3 MiB  =  34×
2048-token prompt,   16 out:  1065 MiB ÷ 0.41 MiB  =  2596×
```

Prefill's share falls with every added prompt token (one weight load spread
over more tokens), so the ratio compounds as the prompt grows. Read the two
columns separately and the reason is plain: **prefill per token falls** while
**decode per token stays flat** (every step reloads everything). That asymmetry
is the hint about the fix.

**Third**, the batching table. Batch size goes 1 → 256, memory traffic stays
**exactly the same**, arithmetic intensity rises 256×:

```
traffic per step  =  2 × params bytes     (weights: loaded once, shared by batch)
flops per step    =  2 × params × batch   (each of B tokens does a full matmul)
intensity         =  2 × params × batch / (2 × params)  =  batch
                 ->  256 / 1  =  256×
```

---

## The idea

The asymmetry comes from one fact:

> Prefill loads the model's weights **once** and uses them for every prompt token.
> Decode reloads **all of them** for every single generated token.

With a 512-token prompt, prefill does 512 tokens' worth of work per weight load.
Decode does one. The weights are ~840 MiB either way.

??? question "Do prefill and decode need *different* weights?"
    No, both read the same 840 MiB. What differs is how many tokens share one
    read: prefill amortizes across 512, decode across 1.

    The question usually comes from conflating **weights** (fixed, 840 MiB,
    re-read every pass) with the **KV cache** (grows per token, written once then
    re-read). Both matter, on different scales, and past ~8k context the KV term
    overtakes the weights.

    [Q&A: why do prefill and decode have different weight requirements?](qa.md#why-do-prefill-and-decode-have-different-weight-requirements)

Both phases read the same 840 MiB out of VRAM. Neither touches the CPU, the
weights were copied to the GPU once at startup and stay there. What repeats is
the VRAM → on-chip transfer, and it repeats *per forward pass*.

This is a **matrix-matrix vs. matrix-vector** distinction. Prefill multiplies a
weight matrix by a matrix of 512 token vectors, lots of arithmetic per byte
fetched. Decode multiplies the same weight matrix by a *single* vector. The GPU
loads 840 MiB to do a rounding error's worth of math, then does it again for the
next token.

So:

- **Prefill is compute-bound.** The GPU's arithmetic units are the limit. Making
  memory faster wouldn't help.
- **Decode is memory-bound.** The GPU is idle, waiting on memory. Making it do
  arithmetic faster wouldn't help at all.

??? question "Then why doesn't a faster GPU make decode faster?"
    TPS is set by arithmetic intensity, not peak FLOPS: decode moves
    bytes/bandwidth, and bandwidth is what you'd have to raise. Doubling FLOPS
    halves TTFT (prefill is compute-bound) and leaves TPS flat — unless the
    extra FLOPS come from batching, which raises intensity instead.

    [Q&A: what happens to TTFT and TPS?](qa.md#you-double-the-flops-but-keep-the-bandwidth-what-happens-to-ttft-and-tps)

**This distinction is the single most useful thing in the book.** Nearly every
optimization ahead is "make decode less memory-bound":

| Technique | How it attacks the bottleneck | Lecture |
|---|---|---|
| KV caching | stop recomputing past keys/values | 05 |
| Batching | one weight load serves N sequences | 07–08 |
| Quantization | make the bytes smaller | 19 |
| Speculative decoding | check several tokens per weight load | 12 |

That third table in the demo is the key. Memory traffic is **fixed**; you were
going to load those weights regardless. Batching gets the extra work for free.
That's not a minor optimization; it's why serving engines exist.

??? question "So is a bigger batch always better?"
    No — it saturates, it never reverses. Past the crossover, throughput
    plateaus at roughly peak_FLOPS / (2 × params) while latency keeps climbing.
    And for this model on a 3090, the crossover only exists for contexts under
    ~100 tokens: realistic decode is memory-bound at every batch size. What
    breaks first is KV capacity, not FLOPS.

    [Q&A: what happens when the batch gets too big?](qa.md#what-happens-when-the-batch-gets-too-big)

### Where it shows up in practice

Request shape decides which phase you're fighting:

- **Summarization**: long prompt, short answer → prefill-heavy, tune TTFT.
- **Chat and agents**: short prompt, long answer → decode-heavy, tune TPS.

They pull in opposite directions, and mixing them on one machine means each
degrades the other. Lecture 11 (chunked prefill) is the first patch for that;
Lecture 27 (disaggregation) is the full answer.

---

## Build it

Nothing to build yet. Do this instead, it takes five minutes and it matters:

1. Create `notes/00-baseline/README.md` (a template is already there).
2. Write down, in your own words, why decode is memory-bound. No looking.
3. **Predict:** on your laptop, generating 256 tokens, what fraction of total
   time is prefill vs. decode? Write the number down. You'll check it in
   Lecture 03.

---

## Go deeper

- **[Field notes](field-notes.md)**: a 2×3090 setup measuring **~1500 tok/s
  prefill against ~100 tok/s decode**. This lecture's asymmetry, as a production
  number. Same source: 100 tok/s for one user, **585 tok/s across 8**, the
  batching table, confirmed.
- **Kiely §2.4.2** (p.63–66), "LLM Inference Bottlenecks." The same split, with
  the memory-movement table this book's Lecture 02 reproduces.
- **Kiely §1.4** (p.35–37), TTFT and TPS as product metrics, not just numbers.
- **vLLM** `vllm/v1/core/sched/scheduler.py`, search for `prefill` and `decode`.
  Don't try to follow it yet; just note that the split is *structural* in real
  engines, not an analysis convenience.

---

## Check yourself

1. Decode does less compute than prefill but takes far longer. Why isn't that a
   contradiction?
2. You double your GPU's FLOPS and keep memory bandwidth the same. What happens
   to TTFT? To TPS?
3. Batching 32 requests costs nearly the same memory traffic as batching 1. Where
   does the extra work go, and what eventually breaks if you keep raising the
   batch size? *(Lecture 09 answers the second half.)*

Answers: [Q1](qa.md#where-does-232960-mib-come-from) ·
[Q2](qa.md#you-double-the-flops-but-keep-the-bandwidth-what-happens-to-ttft-and-tps) ·
[Q3](qa.md#what-happens-when-the-batch-gets-too-big)

---

## Next

**[02. Arithmetic intensity](02-arithmetic-intensity.md)**: make this precise
with numbers you compute yourself.

```bash
uv run python book/code/roofline.py
```

Run it, then read. L02 also has the **KV cache sizing exercise**: the most
practically useful ten minutes in Part I. Don't skip it.

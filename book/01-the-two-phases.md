# 01 — The two phases

**Demo:** `book/code/two_phases.py` · **Moves:** nothing yet — this is the map
**Prereq:** [00 — Introduction](00-intro.md)

---

## The problem

Generating a response looks like one operation. It is two, and they behave so
differently that production systems eventually run them on separate machines.

**Prefill** processes your prompt. All of it, at once, in a single forward pass.
The model computes keys and values for every prompt token in parallel and
produces the first output token. This determines **time to first token (TTFT)** —
how long before the cursor starts moving.

**Decode** generates the rest. One token per forward pass, each depending on the
one before, so there's nothing to parallelize across. Five hundred output tokens
means five hundred sequential passes. This determines **tokens per second (TPS)**
— how fast the text streams.

Same weights, same kernels. Completely different performance characteristics.

---

## See it

```bash
uv run python book/code/two_phases.py
```

Pure arithmetic — no GPU, no model download. Look for three things.

**First**, for a 512-token prompt and 256 tokens out:

```
             compute     memory read    ops:byte
prefill     360.8 GF          672 MiB      512.00
decode      180.4 GF       189952 MiB        0.91
```

Decode does *half* the compute of prefill and moves **283× more memory**.

**Second**, the per-token table. A generated token costs hundreds to thousands of
times the memory traffic of a prompt token — and the ratio *falls* as prompts get
longer, which is a hint about the fix.

**Third**, the batching table. Batch size goes 1 → 256, memory traffic stays
**exactly the same**, arithmetic intensity rises 256×.

---

## The idea

The asymmetry comes from one fact:

> Prefill loads the model's weights **once** and uses them for every prompt token.
> Decode reloads **all of them** for every single generated token.

With a 512-token prompt, prefill does 512 tokens' worth of work per weight load.
Decode does one. The weights are ~672 MiB either way.

This is a **matrix-matrix vs. matrix-vector** distinction. Prefill multiplies a
weight matrix by a matrix of 512 token vectors — lots of arithmetic per byte
fetched. Decode multiplies the same weight matrix by a *single* vector. The GPU
loads 672 MiB to do a rounding error's worth of math, then does it again for the
next token.

So:

- **Prefill is compute-bound.** The GPU's arithmetic units are the limit. Making
  memory faster wouldn't help.
- **Decode is memory-bound.** The GPU is idle, waiting on memory. Making it do
  arithmetic faster wouldn't help at all.

**This distinction is the single most useful thing in the book.** Nearly every
optimization ahead is "make decode less memory-bound":

| Technique | How it attacks the bottleneck | Lecture |
|---|---|---|
| KV caching | stop recomputing past keys/values | 05 |
| Batching | one weight load serves N sequences | 07–08 |
| Quantization | make the bytes smaller | 19 |
| Speculative decoding | check several tokens per weight load | 12 |

That third table in the demo is the key. Memory traffic is **fixed** — you were
going to load those weights regardless. Batching gets the extra work for free.
That's not a minor optimization; it's why serving engines exist.

### Where it shows up in practice

Request shape decides which phase you're fighting:

- **Summarization** — long prompt, short answer → prefill-heavy, tune TTFT.
- **Chat and agents** — short prompt, long answer → decode-heavy, tune TPS.

They pull in opposite directions, and mixing them on one machine means each
degrades the other. Lecture 11 (chunked prefill) is the first patch for that;
Lecture 27 (disaggregation) is the full answer.

---

## Build it

Nothing to build yet. Do this instead — it takes five minutes and it matters:

1. Create `notes/00-baseline/README.md` (a template is already there).
2. Write down, in your own words, why decode is memory-bound. No looking.
3. **Predict:** on your laptop, generating 256 tokens — what fraction of total
   time is prefill vs. decode? Write the number down. You'll check it in
   Lecture 03.

---

## Go deeper

- **Kiely §2.4.2** (p.63–66) — "LLM Inference Bottlenecks." The same split, with
  the memory-movement table this book's Lecture 02 reproduces.
- **Kiely §1.4** (p.35–37) — TTFT and TPS as product metrics, not just numbers.
- **vLLM** `vllm/v1/core/sched/scheduler.py` — search for `prefill` and `decode`.
  Don't try to follow it yet; just note that the split is *structural* in real
  engines, not an analysis convenience.

---

## Check yourself

1. Decode does less compute than prefill but takes far longer. Why isn't that a
   contradiction?
2. You double your GPU's FLOPS and keep memory bandwidth the same. What happens
   to TTFT? To TPS?
3. Batching 32 requests costs nearly the same memory traffic as batching 1. Where
   does the extra work go — and what eventually breaks if you keep raising the
   batch size? *(Lecture 09 answers the second half.)*

---

**Next:** [02 — Arithmetic intensity](02-arithmetic-intensity.md) — make this
precise with numbers you compute yourself.

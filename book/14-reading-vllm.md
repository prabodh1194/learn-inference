# 14 — Reading vLLM

**Build:** nothing · **Test:** none · **Moves:** your ability to read production code
**Prereq:** [13 — CUDA graphs](13-cuda-graphs.md) — and a working engine

---

## Why now and not earlier

You have built: a KV cache, a scheduler doing continuous batching, a paged block
manager with prefix caching, chunked prefill, and speculative decoding.

Every one of those is a thing vLLM does. You now have **opinions** about them —
you know why the parent hash goes in the block hash, why retire comes before
admit, why chunk size trades p99 against throughput.

That's the difference this lecture depends on:

> Read *Inside vLLM* before building a scheduler and it teaches you vLLM's
> **vocabulary**. Read it after and it teaches you vLLM's **judgment**.

You're not here to learn what continuous batching is. You're here to see what a
team does with it after four years, thousands of users, and constraints you
haven't hit yet.

---

## Read in this order

### 1. Gordić, *Inside vLLM* — the whole thing

**[aleksagordic.com/blog/vllm](https://www.aleksagordic.com/blog/vllm)**

~7,500 words, top-down through vLLM V1. Every heading is something you built:

| Section | You built it in |
|---|---|
| LLM Engine & Engine Core | L08 |
| Scheduler | L08 |
| Chunked prefill | L11 |
| Prefix caching | L10 |
| Guided Decoding (FSM) | L12b |
| Speculative decoding | L12 |
| Disaggregated P/D | L27 (preview) |
| MultiProcExecutor | L22 (preview) |

Read it in one sitting. **Keep a list of every place vLLM does something
differently than you did**, and don't resolve them yet — just collect.

### 2. nano-vllm — read the source

**[github.com/GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)**
— ~1,200 lines, MIT. The nanoGPT of inference: the same architecture as vLLM with
the production complexity removed.

One sitting per file, in this order:

| File | ~Size | Compare against |
|---|---|---|
| `engine/sequence.py` | 2.6K | your `Sequence` |
| `engine/scheduler.py` | 3.7K | your `Scheduler` (L08) |
| `engine/block_manager.py` | 4.3K | your `BlockManager` (L09, L10) |
| `engine/model_runner.py` | 12K | your runner (L13) |
| `layers/attention.py` | 2.8K | your paged gather |

For each: read it, then **diff it against yours mentally**. Where it differs, ask
whether they're handling a case you didn't, or making a different tradeoff.

### 3. vLLM itself — targeted, not exhaustive

Do **not** read vLLM top to bottom. It's hundreds of thousands of lines with
multi-backend support, quantization schemes, and hardware variants. Read these
four files, each against the lecture that taught it:

| File | Lecture |
|---|---|
| `vllm/v1/core/sched/scheduler.py` | L08, L11 |
| `vllm/v1/core/block_pool.py` | L09 |
| `vllm/v1/core/kv_cache_manager.py` | L09 |
| `vllm/v1/core/kv_cache_utils.py` | L10 — block hashing |

In `kv_cache_utils.py`, look closely at what goes **into** the block hash. LoRA
adapter id, multimodal inputs, cache salt. Every one of those is a correctness
bug someone shipped: two requests with identical tokens but different *context*
must not share blocks. You learned the principle in L10; here's the accumulated
scar tissue.

---

## Questions to answer from the source

Answer these in `notes/01-engine/README.md`, citing file and function:

1. **Preemption.** When vLLM runs out of blocks, how does it choose a victim, and
   does it swap or recompute? What decides?
2. **Watermark.** vLLM keeps some blocks in reserve rather than allocating to
   zero. Why? What breaks without it?
3. **Scheduler budget.** How does vLLM account for a step's cost? Compare to your
   `max_batched_tokens`.
4. **Block hash inputs.** List everything included. Which surprised you, and what
   bug does each prevent?
5. **Cascade attention.** Find where vLLM optimizes attention for a batch sharing
   a long prefix. Why is that a special case worth its own path?

---

## Now compare honestly

Write a diff-style entry in your notes: **where does your design differ, and were
they right?**

Some will be "they handle a case I ignored" — multi-modal inputs, LoRA, preemption
under sustained pressure. Some will be "they made a different tradeoff" — block
size defaults, eviction policy, when to give up on speculation.

And some of your choices will be *fine*. A simpler design that handles your cases
isn't worse; production complexity is mostly the cost of generality you don't
need. Recognizing which is which is the skill this lecture is for.

---

## Optional: SGLang

**[github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)** takes a
different path on prefix caching — **RadixAttention**, a radix tree rather than a
flat hash map, so prefixes share *structurally*. Better for branching
conversation trees where many requests share nested prefixes.

Read [the RadixAttention paper](https://arxiv.org/abs/2312.07104) §3 and ask why
you'd choose one over the other. That's a genuine engineering question with no
universal answer.

---

## Check yourself

You should now be able to:

1. Open any file in `vllm/v1/core/` and explain what it's for.
2. Name three things vLLM does that you don't, and why each exists.
3. Name one thing you'd do differently, and defend it.
4. Read a vLLM issue about throughput and form an opinion about the cause.

If (4) feels reachable, Part II worked.

---

**Part II complete.** You have an engine, and the real ones are no longer opaque.

## Next

**[15 — Profiling](15-profiling.md)** — Part III goes down a level, to where the
time actually goes.

> **Part III needs an NVIDIA GPU for every lecture.** Nsight is CUDA-only. This
> is the point to rent one and work through L15–L20 in a block.

Its deliverable is a **ranked kernel table**. Every optimization in L16–L20 must
cite it — you optimize the top row, not the interesting one.

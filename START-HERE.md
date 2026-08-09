# Start here

A numbered path through the course. Do them in order. Check off as you go.

Anything marked **🖥️ GPU** needs a rented NVIDIA box — everything else runs on
your laptop.

---

## Setup (15 minutes, once)

**1.** Install dependencies.

```bash
cd ~/personal/learn-inference
uv sync --group dev
```

**2.** Prove the environment works. This runs today, no model, no GPU:

```bash
uv run python book/code/roofline.py
```

You should see `intensity 62.4 ops:byte (book says ~62)`. If you do, everything
downstream will work.

**3.** Download the model (~1.2GB, once).

```bash
uv run python scripts/fetch_model.py
```

**4.** See where you are. You'll use this command constantly.

```bash
uv run python scripts/progress.py
```

Failures are your homework, not breakage. `[!]` is the only status that means
something is wrong.

---

## Part I — Foundations *(laptop, ~1 evening)*

**5.** Read [`book/00-intro.md`](book/00-intro.md). Ten minutes. It explains why
inference is its own discipline and why training intuitions mislead.

**6.** Read [`book/01-the-two-phases.md`](book/01-the-two-phases.md), and run its
demo when it tells you to:

```bash
uv run python book/code/two_phases.py
```

The number to notice: decode moves **277× more memory** than prefill for half
the compute.

**7.** Read [`book/02-arithmetic-intensity.md`](book/02-arithmetic-intensity.md).
You already ran its demo in step 2 — read the output again now that the lecture
explains it.

**8.** **Do the sizing exercise** at the end of L02 (KV cache for one 4096-token
sequence; how many fit on a 24GB card; then redo it without GQA). Write the
answers in `notes/00-baseline/README.md`.

> This is the most common practical question in inference work. Don't skip it.

**9.** Read [`book/03-naive-generation.md`](book/03-naive-generation.md) and run:

```bash
uv run python book/code/recomputation.py
```

**99.7% of the K/V work is thrown away.** That's what you're about to fix.

---

## 🔨 Your first code

**10.** Write your prediction in `notes/00-baseline/README.md` **before** coding:
*per-token latency vs. position — flat, linear, or quadratic?*

**11.** Implement two functions (~30 lines total):

- `engine/model.py::load` — load Qwen3-0.6B, eval mode, MPS, **float32**
- `engine/generate.py::generate_naive` — the loop from L03

L03 has the sketch. Use float32 on MPS to start; fp16 has accuracy quirks that
break the test for reasons unrelated to your loop.

**12.** Prove it's correct:

```bash
uv run pytest tests/test_03_generation.py -v
```

Your greedy output must match HuggingFace **exactly**. Greedy is deterministic,
so a mismatch is a real bug — most often sampling the wrong logit position
(want `logits[:, -1]`).

**13.** Measure it. This is the payoff:

```bash
uv run python book/code/naive_bench.py
```

**14.** Compare the plot to your step-10 prediction. Write down what surprised
you — *especially if you were wrong*. That entry is worth more than the code.

**15.** Read [`book/04-measuring.md`](book/04-measuring.md) and do its
`synchronize()` experiment. Seeing a fake 100× speedup yourself is worth more
than being told it's possible.

✅ **Gate:** you can state, from your own numbers, why decode is memory-bound and
prefill is compute-bound.

---

## Part II — The engine *(laptop through step 21)*

Each lecture is the same loop: **read → run the demo → predict → implement →
test → measure → record**.

**16.** [L05 — KV cache](book/05-kv-cache.md) → `generate_cached`
→ `pytest tests/test_05_kv_cache.py`
→ `uv run python book/code/naive_bench.py --cached`

*The curve flattens. This is the most satisfying moment in Part II.*

**17.** [L06 — Sampling](book/06-sampling.md) → `engine/sampling.py`
Short. It's what makes every later test trustworthy.

**18.** [L07 — Static batching](book/07-static-batching.md) → `generate_batched`
→ `uv run python book/code/batch_bench.py`
*Record the padding waste, not just throughput.*

**19.** [L08 — Continuous batching](book/08-continuous-batching.md) →
`engine/scheduler.py`

> The big one. `generate()` turns inside out into scheduler + runner. A real
> refactor — budget more time. Its 10 tests need no model, so iterate fast.

**20.** [L09 — Paged attention](book/09-paged-attention.md) →
`engine/block_manager.py`
Logic is testable on the laptop; **🖥️ GPU** for the capacity measurement.

**21.** [L10 — Prefix caching](book/10-prefix-caching.md) → `match_prefix`
→ `uv run python book/code/prefix_bench.py`
*Run `shared_prefix` and `late_divergence` back to back. Same tokens, opposite
results.*

**22.** [L11 — Chunked prefill](book/11-chunked-prefill.md) — judge on **p99**.
The mean barely moves; watching the mean concludes you did nothing.

**23.** [L12 — Speculative decoding](book/12-speculative-decoding.md) → n-gram
draft/verify. *Report acceptance rate, always.*

**24.** [L12b — Structured output](book/12b-structured-output.md) → the logit
hook. Guided decoding, tool calling, LoRA.

**25.** 🖥️ [L13 — CUDA graphs](book/13-cuda-graphs.md) — when Python, not the
GPU, is the bottleneck.

**26.** **[L14 — Reading vLLM](book/14-reading-vllm.md)** — the capstone.
*Now* read [Gordić's post](https://www.aleksagordic.com/blog/vllm), then
nano-vllm file by file.

> Read it before this point and it teaches vocabulary. Read it here and it
> teaches judgment. That's why it's step 26 and not step 5.

✅ **Gate:** you have a working engine and can read vLLM without it being magic.
**This is the "already useful" checkpoint** — everything after is depth.

---

## Part III — Kernels 🖥️ *(rent a 3090, ~$0.25/hr)*

**27.** [L15 — Profiling](book/15-profiling.md). **Build your kernel ranking
table first.** Every later optimization must cite it.

```bash
uv run python -c "from kernels.profile_engine import report; report({'linear':38,'attn':18,'norm':7})"
```

**28.** [L16 — Triton basics](book/16-triton-basics.md) — softmax, RMSNorm.
**29.** [L17 — FlashAttention](book/17-flash-attention.md) — the big one.
**30.** [L18 — Paged attention kernel](book/18-paged-attention-kernel.md).
**31.** [L19 — Quantization](book/19-quantization.md) — **build the quality
harness before you benchmark speed.**
**32.** [L20 — Raw CUDA](book/20-raw-cuda.md) — you'll probably lose to Triton.
That's the lesson.

---

## Part IV — Parallelism 🖥️ *(2+ GPUs for step 34)*

**33.** [L21 — JAX and XLA](book/21-jax-and-xla.md).
**34.** [L22 — Tensor parallelism](book/22-tensor-parallelism.md) — declare it,
find the all-reduce XLA inserted, then write it yourself.
**35.** [L23 — MoE](book/23-moe-and-expert-parallelism.md).

---

## Part V — Production

**36.** [L24 — Serving](book/24-serving.md) — OpenAI-compatible API.
**37.** [L25 — Load testing](book/25-load-testing.md) — find your knee.
**38.** [L26 — Versus vLLM](book/26-versus-vllm.md) — **you will lose.** Explain
the gap with a profiler. Most educational benchmark in the book.
**39.** [L27 — Routing and disaggregation](book/27-routing-and-disaggregation.md).
**40.** [L28 — Autoscaling and cost](book/28-autoscaling-and-cost.md):

```bash
uv run python bench/cost_model.py
```

*Idling at 20% costs more than doubling your batch size saves.*

**41.** [L29 — Contributing](book/29-contributing.md) — go land a PR.

---

## The three habits

Worth more than any single lecture:

**Predict before you measure.** Wrong predictions, written down and left
unedited, are how the intuition actually forms.

**Match the workload to the claim.** Continuous batching looks worthless on
uniform load; speculative decoding looks like magic on code and useless on
prose. A benchmark without a stated workload says nothing.

**Keep the failures.** The kernel that came out slower, the optimization that did
nothing. Those `notes/` entries are the record of your model of the machine being
corrected — which is the whole point.

---

## If you get stuck

```bash
uv run python scripts/progress.py                      # where am I?
uv run pytest tests/test_05_kv_cache.py -v             # what exactly is failing?
uv run pytest -m "not cuda" -q | grep NotImplementedError | sort -u
```

Reference: [`book/README.md`](book/README.md) (full TOC) ·
[`book/field-notes.md`](book/field-notes.md) (what practitioners actually
measure) · [`book/appendix-a-glossary.md`](book/appendix-a-glossary.md) ·
[`tests/README.md`](tests/README.md)

**Right now: step 1.**

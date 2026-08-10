# 00 — Introduction

**Build:** the environment · **Test:** `uv run python scripts/progress.py`
**Prereq:** you can read PyTorch and know what attention computes

---

## Why inference is its own discipline

This book assumes you can read PyTorch and know roughly what attention computes
— Q, K, V, and why it's causal. It does **not** assume you've trained a model,
and it never asks you to.

If you have trained one, one habit needs unlearning, and it's the reason this
lecture leads with it: **inference optimizes for something training never cares
about — one token, right now.**

Training is **throughput-only**. Nobody is waiting on any individual step, so
you're free to make batches as large as memory allows. Every sequence in a batch
is processed in parallel, which means the matrix multiplies are large and the
weights you load get used for a great deal of arithmetic.

Serving is the opposite. Requests arrive when they arrive, at wildly different
lengths. Someone is watching a cursor blink. You generate **one token at a time**,
each depending on the last, so there's no parallelism *along* the sequence to
exploit — and the same matrix multiply that was large and efficient during
training becomes a matrix-*vector* product that barely uses the hardware.

> **Two honest caveats**, since this comparison is doing a lot of work.
>
> Training isn't purely throughput-bound in practice — gradient synchronization,
> optimizer state, and activation memory all impose their own limits, and real
> pipelines use sequence packing rather than the naive padding this contrast
> implies. The claim that survives all of that is narrower and is the only one
> this book relies on: **training processes many tokens per weight load; decode
> processes one.**
>
> And note what is *deliberately* absent above: any claim about "GPU
> utilization." That metric is misleading here — a memory-bound decode loop can
> report high utilization while doing very little useful work. Lecture 15 shows
> why, and Lecture 28 explains why you must not autoscale on it. We'll use
> arithmetic intensity instead, which doesn't have that failure mode.

That last part is the crux, and it's worth stating precisely because it drives
nearly everything in this book:

> Generating one token requires reading **every weight in the model** out of GPU
> memory and doing almost no arithmetic with them.

**Which memory matters here.** The weights already live in **VRAM** — they were
copied there once at startup. The traffic that costs you is VRAM → the GPU's
on-chip SRAM and registers, and it happens on *every forward pass*, because
there is nowhere near enough on-chip memory to hold 840 MiB of weights.

| Path | Bandwidth (RTX 3090) | When |
|---|---|---|
| **VRAM → on-chip** | ~936 GB/s | **every step** — this is the bottleneck |
| CPU RAM → VRAM (PCIe) | ~64 GB/s | once at load time |

If weights crossed PCIe every step you'd be another ~15× slower. When that
*does* happen it's called offloading, and it's what you resort to when a model
doesn't fit — not how normal serving works.

A matrix-*matrix* multiply over a batch does lots of work per byte loaded. A
matrix-*vector* multiply for a single token does almost none. Same weights, same
kernel, wildly different efficiency. In Lecture 02 you'll compute this exactly:
Qwen3-0.6B decode runs at **0.75 operations per byte**, against an H100 that
needs 295 to keep its arithmetic units busy. That's roughly 0.25% — and the
conclusion isn't hardware-specific: the same calculation puts decode far to the
memory-bound side on an A100, a 3090, and an M1 alike.

Most of Part II answers one question: *"how do we get more work out of each byte
we were going to load anyway?"*

- **Batching** — load the weights once, generate for 32 sequences instead of 1.
- **KV caching** — don't recompute what you already computed.
- **Quantization** — make the bytes smaller.
- **Speculative decoding** — verify several tokens in the time one would take.

Four techniques, one bottleneck. Once you see it, that part of the field stops
being a list of tricks and becomes a single idea with variations.

It is not the *only* idea, and this book doesn't pretend otherwise. Paged
attention (L09) attacks **memory capacity**, not bandwidth. CUDA graphs (L13)
attack **CPU launch overhead** — cases where the GPU isn't the bottleneck at all.
Chunked prefill (L11) redistributes work without reducing it. And Part V is about
utilization and cost, where the biggest lever is often not touching the engine.

---

## How this book works

The method is borrowed from Karpathy's Zero-to-Hero (no affiliation — it's
simply the format that works), aimed at a different target:

> **build the naive thing → measure it → find the bottleneck → fix it → measure again**

You will write a slow generation loop on purpose, plot how badly it scales, and
only then build the KV cache. That order is deliberate. Reading "the KV cache
avoids recomputation" teaches you a sentence; watching your own per-token latency
climb and then flatten teaches you the thing.

**The measurement is not optional.** It's what separates understanding from
trivia. Every lecture names a number that must move, and `bench/` exists to
produce it.

### The repo

```
book/     lectures + runnable demos   READ
engine/   stubs you fill in           BUILD
bench/    measurement harness         MEASURE
tests/    correctness gates           PROVE
notes/    your results and surprises  RECORD
```

`tests/` is what makes progress unambiguous. Every implementation milestone has a
test asserting your version matches a reference — your greedy output must match
HuggingFace's exactly, your paged attention must match contiguous attention, your
Triton kernel must match PyTorch.

Green means **correct on the cases tested**, which is weaker than "done" but far
stronger than "seems fine" — and it's what lets you optimize aggressively later,
because you find out immediately when speed costs you correctness.

### Notes

Keep a lab journal in `notes/`. Two rules, both from experience:

**Write your prediction before you run anything.** Every lecture that can be
predicted asks you to. Wrong predictions, left unedited, are the highest-value
thing in the repo — they're the record of a wrong model of the machine being
corrected.

**Keep the failures.** The optimization that did nothing. The kernel that came out
slower. These are not embarrassing; they're most of what expertise actually is.

---

## What you'll build

By Lecture 14 you'll have an engine that does continuous batching over a paged KV
cache, with prefix caching and speculative decoding.

Those are the same *ideas* vLLM is built on, and enough shared structure — a
scheduler/runner split, a block manager, a block table — that its source becomes
readable. It is **not** the same system: vLLM has multi-backend support, dozens
of quantization schemes, hardware-specific kernels, multimodal inputs, LoRA, and
years of production edge cases. Expect to lose to it decisively in Lecture 26,
and to be able to explain exactly where and why.

Parts III–V go down (Triton and CUDA kernels), sideways (JAX, tensor parallelism),
and out (serving, load testing, cost).

---

## Setup

```bash
git clone https://github.com/prabodh1194/learn-inference
cd learn-inference
uv venv --python 3.12
uv sync --group dev
```

Check it works — this runs today, with no GPU and no model download:

```bash
uv run python book/code/roofline.py
uv run pytest -m "not cuda" -q
```

**The test suite will report a lot of failures. That is the correct starting
state**, not a broken checkout. Every failure is a `NotImplementedError` naming
the lecture that fills it in — the suite is a specification, and you turn it
green as you go.

```bash
uv run python scripts/progress.py     # the readable version
```

Roughly 58 tests pass on a fresh clone; those pin arithmetic you can check
before writing any code (the roofline derivation, cost models, sharding math).

The first command prints a **roofline** analysis. If that word means nothing
yet, that's fine and expected — Lecture 02 derives it properly. The one-line
version, so the output isn't opaque:

> A GPU has two ceilings — how fast it can compute, and how fast it can read
> memory. The roofline compares them, and tells you which one an operation is
> actually stuck against.

The number to look for in that output is **0.75 ops:byte** for decode, against a
ceiling that only stops mattering above ~295. That gap is the reason this book
exists.

### Hardware

Parts I and II through Lecture 08 run on a laptop, using PyTorch's MPS backend
on Apple silicon. Correctness, scheduling logic, and the shape of every curve are
all visible there.

**On 8GB it is tight but workable.** Qwen3-0.6B is 2.2 GiB in float32 (which
Lecture 03 recommends on MPS, because fp16 has accuracy quirks there) against
8GB of *unified* memory shared with the OS. If you hit memory pressure, switch
to bfloat16 — halving the weights to 1.1 GiB — and re-check the Lecture 03
correctness test still passes before trusting any later numbers.

From Lecture 09 you want a real NVIDIA GPU — paged attention and CUDA graphs are
CUDA-specific, and Part III's profiling requires Nsight.

A **24GB card** is the sweet spot: enough VRAM to make the memory lectures real,
without paying datacenter prices. From [Vast.ai's pricing
page](https://vast.ai/pricing) (checked while writing this — **verify before you
rent**, these move):

| GPU | VRAM | from | median |
|---|---|---|---|
| **RTX 3090** (Ampere) | 24GB | $0.05/hr | **$0.16/hr** |
| RTX 4090 (Ada) | 24GB | $0.13/hr | $0.36/hr |
| H100 NVL (Hopper) | 80GB | $1.53/hr | $2.33/hr |

The 3090 is the recommendation: cheapest of the three, and 24GB is plenty for
Qwen3-0.6B. Its one gap is no FP8, which affects exactly one milestone in
Lecture 19 — INT8 works fine there, and the 3090 has hardware INT4 besides.

Treat the "from" column with suspicion: it's the cheapest listing on the
marketplace, often an unreliable host or a bad location. **Median is the number
to plan against.** Even so, everything in this book is a few dollars of GPU
time, not hundreds.

Rent in blocks, prepare your code offline, and **always stop the pod**.

### The model

**Qwen3-0.6B** throughout. Small enough to iterate on a laptop, real enough to be
non-trivial — grouped-query attention, RoPE, RMSNorm, SwiGLU. Using one model
start to finish means every benchmark you take is comparable to every other.

```bash
uv run python scripts/fetch_model.py
```

---

## Go deeper

- Kiely, *Inference Engineering*, Preface and Ch. 0 — why inference became the
  industry's center of gravity.
- Kiely §1.1–1.2 (p.26–30) — where inference work sits in a product.

---

## Check yourself

Before Lecture 01, you should be able to say:

1. Why does a technique that helps training throughput not necessarily help
   serving latency?
2. Decode reads every weight to produce one token. Why does batching 32 requests
   cost barely more time than batching 1?
3. Those weights are read *from where, to where* — and how often? What would be
   true if they came from CPU RAM instead?

If (2) isn't obvious yet, good — that's Lecture 01. (3) is a common slip and
worth getting exactly right before you go on.

---

**Next:** [01 — The two phases](01-the-two-phases.md)

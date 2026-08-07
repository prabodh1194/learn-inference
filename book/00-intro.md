# 00 — Introduction

**Build:** the environment · **Test:** `pytest -m "not cuda"` · **Prereq:** you've trained a transformer

---

## Why inference is its own discipline

You know how a transformer works. You've implemented attention, trained a model,
watched a loss curve go down. That knowledge is necessary here and it is not
sufficient, because **inference optimizes for something training never cares
about: one token, right now.**

Training is throughput-only. You have a giant batch, every sequence is the same
length because you padded them, the GPU is saturated, and nobody is waiting. If a
step takes 400ms instead of 380ms, you shrug.

Serving is the opposite. Requests arrive when they arrive, at wildly different
lengths. Someone is watching a cursor blink. You generate **one token at a time**,
each depending on the last, so there's no way to parallelize across the sequence
— and the GPU that was 90% utilized during training now sits mostly idle,
waiting on memory.

That last part is the crux, and it's worth stating precisely because it drives
nearly everything in this book:

> Generating one token requires reading **every weight in the model** from memory
> and doing almost no arithmetic with them.

A matrix-*matrix* multiply over a batch does lots of work per byte loaded. A
matrix-*vector* multiply for a single token does almost none. Same weights, same
kernel, wildly different efficiency. In Lecture 02 you'll compute this exactly:
Qwen3-0.6B decode runs at **0.75 operations per byte** on a machine that can
sustain 295. You are using roughly 0.25% of the arithmetic the GPU can do.

Nearly every technique in this book is a different answer to *"how do we get more
work out of each byte we were going to load anyway?"*

- **Batching** — load the weights once, generate for 32 sequences instead of 1.
- **KV caching** — don't recompute what you already computed.
- **Quantization** — make the bytes smaller.
- **Speculative decoding** — verify several tokens in the time one would take.

Four techniques, one bottleneck. Once you see it, the field stops being a list of
tricks and becomes a single idea with variations.

---

## How this book works

Same method as Karpathy's Zero-to-Hero, aimed at a different target:

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
Triton kernel must match PyTorch. Green means done, not "seems fine."

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
cache with prefix caching and speculative decoding — the same architecture as
vLLM, smaller and slower but genuinely the same ideas. Then you read vLLM and
find it comprehensible.

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

You should see the roofline numbers print, and the test suite pass with a few
skips (those are the tests waiting on code you haven't written yet — that's the
correct starting state).

### Hardware

Parts I and II through Lecture 08 run fine on a laptop. **An Apple M1 with 8GB is
enough**, using PyTorch's MPS backend. Correctness, scheduling logic, and the
shape of every curve are all visible there.

From Lecture 09 you want a real NVIDIA GPU — paged attention and CUDA graphs are
CUDA-specific, and Part III's profiling requires Nsight. An **RTX 3090 on Vast.ai
runs about $0.20–0.25/hour**, which is a few dollars for everything in this book.
Rent in blocks, prepare offline, and always stop the pod.

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

If (2) isn't obvious yet, good — that's Lecture 01.

---

**Next:** [01 — The two phases](01-the-two-phases.md)

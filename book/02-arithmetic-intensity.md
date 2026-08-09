# 02 — Arithmetic intensity

**Demo:** `book/code/roofline.py` · **Test:** `tests/test_02_roofline.py`
**Moves:** nothing — this is how you *predict* what will move · **Prereq:** [01](01-the-two-phases.md)

---

## The problem

Lecture 01 said decode is memory-bound. That was an argument. This lecture makes
it a **number**, so you can predict whether an optimization will help *before*
spending a weekend on it.

The question you want to answer about any operation: **is the GPU waiting on
arithmetic, or waiting on memory?** Because if it's waiting on memory, a faster
kernel buys you nothing.

---

## The idea

Two numbers describe a GPU:

- **Peak compute** — floating-point operations per second.
- **Peak bandwidth** — bytes per second from memory.

Divide them and you get the **ops:byte ratio** — how much arithmetic the machine
must do per byte loaded to keep its compute units busy. For an H100: 989 TFLOPS ÷
3.35 TB/s ≈ **295 operations per byte**. Fetch a byte, do fewer than 295
operations with it, and you've wasted the trip.

> **Which FLOPS number?** Spec sheets list several, differing by up to 8×, and
> picking the wrong one silently corrupts every prediction you make.
>
> TechPowerUp lists the RTX 3090 at "FP16 (half) 35.58 TFLOPS **(1:1)**" — that's
> the *shader* rate, where fp16 runs no faster than fp32. Matmuls don't use
> shaders; they use **tensor cores**, and the 3090's dense fp16 tensor rate is
> ~71 TFLOPS. That's the number in `roofline.py`.
>
> Vendors also quote **sparse** rates, which are 2× dense and require 2:4
> structured sparsity you almost certainly don't have. If a headline number looks
> suspiciously round and large, check whether it's sparse.
>
> Rule: **dense tensor-core rate, at the dtype you actually run.**

Now the same measure for an *algorithm*, called **arithmetic intensity**:

```
intensity = total compute (ops) / total memory traffic (bytes)
```

Compare the two and you have your answer:

- intensity **>** ops:byte → **compute-bound** (arithmetic is the limit)
- intensity **<** ops:byte → **memory-bound** (bandwidth is the limit)

### The roofline

Plot those two facts together and you get the chart the lecture is named after.
It's called a roofline because of its shape — a diagonal rising into a flat,
like a roof:

```
  performance
  (FLOP/s)
      ^
      |              ______________________  <- compute ceiling
      |             /                           (peak FLOPS)
      |            /
      |           /   ← compute-bound: adding
      |          /      bandwidth won't help
      |         /
      |        /  <- bandwidth ceiling
      |       /      (slope = bytes/s)
      |      /
      |     /  ← memory-bound: adding
      |    /     FLOPS won't help
      |   /
      +--+--------|-------------------------> arithmetic intensity
                the ridge                        (FLOP per byte)
             (= ops:byte ratio)
```

Two ceilings, because the machine has two limits:

- **The diagonal** — you cannot compute faster than memory can feed you. Its
  slope *is* the memory bandwidth.
- **The horizontal** — you cannot exceed the arithmetic units. Its height *is*
  peak FLOPS.

The corner where they meet is the **ridge point**, and it sits at exactly the
ops:byte ratio (295 for an H100, 76 for a 3090). That's the whole reason the
ratio matters: it's the intensity at which the two ceilings cross.

Now locate your algorithm on the x-axis by its arithmetic intensity:

- **Left of the ridge → memory-bound.** You're on the diagonal. A faster GPU
  changes nothing; more bandwidth is the only thing that helps.
- **Right of the ridge → compute-bound.** You're on the flat. More bandwidth
  changes nothing; more FLOPS is the only thing that helps.

That's the entire tool. One number (your intensity), one comparison (against the
ridge), and you know which half of the hardware you're wasting.

**Why this book is organized around it:** decode lands at **0.75 ops:byte**
against a ridge of 295 — roughly 400× to the left, using about a quarter of one
percent of the GPU's arithmetic. Prefill lands at ~510, on the other side. Same
weights, same kernels, opposite ceilings.

Every technique in Parts II and III is an answer to the same question: *how do I
get more arithmetic out of bytes I was going to load anyway?* Batching, KV
caching, quantization, and speculative decoding are four different answers, and
the roofline is what tells you they're all the same idea.

### Doing it for attention

Take unoptimized attention — `S = QK^T`, `P = softmax(S)`, `O = PV` — with
sequence length `N`, head dim `d`, FP16 (2 bytes/value). Each step reads from
memory, computes, writes back. Add it up:

```
memory  = 8N² + 8Nd   bytes
compute = 4N²d + 3N²  ops
```

At N=4096, d=128, that's **62 ops:byte**. Against an H100's 295, attention is
memory-bound by nearly 5×.

Notice *why*: those `N²` terms are the score matrix, written to memory and
immediately read back — 32 MiB at this size, round-tripped for nothing. Deleting
that round-trip is exactly what FlashAttention does (Lecture 17).

---

## See it

```bash
uv run python book/code/roofline.py
```

Confirm the book's worked example first:

```
intensity     62.4 ops:byte   (book says ~62)
```

Then the number that matters:

```
decode  (1 token)          0.75 ops:byte
prefill ( 512 tokens)    510.48 ops:byte
```

**0.75 against a ridge of 295.** Decode uses roughly a quarter of one percent of
the arithmetic the machine can do. It is not a little memory-bound; it is almost
entirely memory-bound.

And the verdict table shows this holds on *every* device — H100 (ridge 295), A100
(153), 3090 (76), M1 (38). A conclusion that survives an 8× range of ridge points is a
property of the algorithm, not a quirk of one GPU.

One aside worth noticing: the **3090's ridge is 76, the H100's is 295.** The
cheaper card has proportionally *more* bandwidth per FLOP, making it relatively
better at memory-bound decode. "Slower GPU" and "worse at decode" are not the
same claim.

---

## Build it

1. Run the demo. Check 62.4 against Kiely Fig 2.18 (p.66).
2. Run `uv run pytest tests/test_02_roofline.py -v` — these pass today. Read
   them; they encode the claims above as assertions.
3. **Fill in your own hardware.** The `Device` entries are nominal spec-sheet
   figures. Find your laptop's real numbers and add them.
4. **The exercise that matters — KV cache sizing.** Kiely §5.4 (Fig 5.11, p.142)
   gives the formula for VRAM. Using `ModelDims.kv_bytes_per_token()`:
   - How much KV cache does one 4096-token sequence need?
   - On a 24GB 3090 with ~840 MiB of weights (what `roofline.py` prints), how
     many such sequences fit?
   - Now recompute with `n_kv_heads=16` instead of 8 (i.e. no GQA). How many fit?

   That last comparison is why grouped-query attention exists, and "how big is
   the KV cache" is the most common practical question in this whole field.

Record the answers in `notes/00-baseline/README.md`.

---

## Go deeper

- **Kiely §2.4–2.4.2** (p.61–66) — the derivation this lecture reproduces.
- **Kiely §2.5** (p.67–70) — how FlashAttention and PagedAttention each attack
  the numbers you just computed.
- **[Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785)**
  (Williams et al., 2009) — the original. Predates GPUs in this role and still
  the clearest statement of the idea.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)** (Dao et al., 2022) —
  §2 has the memory-traffic analysis. Skim now, implement in Lecture 17.

---

## Check yourself

Answer from your own output:

1. Decode is 0.75 ops:byte and the H100's ridge is 295. If you doubled that GPU's
   FLOPS, how much faster does decode get?
2. Attention intensity rises with N (32 at N=128, 62 at N=4096) but never passes
   the ridge. What does that tell you about attention at *any* sequence length?
3. Your predicted batch-size ceiling from the sizing exercise — what runs out
   first, and what would you change to raise it? *(Lectures 09 and 19.)*

---

**Next:** [03 — Naive generation](03-naive-generation.md) — stop predicting,
start measuring.

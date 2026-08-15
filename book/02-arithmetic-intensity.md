# 02. Arithmetic intensity

**Demo:** `book/code/roofline.py` · **Test:** `tests/test_02_roofline.py`
**Moves:** nothing: this is how you *predict* what will move · **Prereq:** [01](01-the-two-phases.md)

---

## The problem

Lecture 01 said decode is memory-bound. That was an argument. This lecture makes
it a **number**, so you can predict whether an optimization will help *before*
spending a weekend on it.

The question you want to answer about any operation: **is the GPU waiting on
arithmetic, or waiting on memory?** Because if it's waiting on memory, a faster
kernel buys you nothing.

Think of a kitchen. The chef's knife is the GPU's arithmetic: so many chops
per second, however fast the blade is. The fridge is the GPU's memory: it
hands over a fixed number of ingredients per second, no matter what. Every
dish is fetch ingredients, chop, fetch more, chop, and the question the
roofline answers is which of the two you spend most of your time waiting on.
If you're always walking to the fridge, buying a faster knife changes
nothing.

---

## The idea

Two numbers describe a GPU:

- **Peak compute**: floating-point operations per second. A floating-point
  operation (FLOP) is one multiply or one add; TFLOPS is trillions of them
  per second.
- **Peak bandwidth**: bytes per second from memory. The fridge trip: the
  ceiling on how much data can arrive, however fast the chips can chew it.

Divide them and you get the **ops:byte ratio**, how much arithmetic the machine
must do per byte loaded to keep its compute units busy. For an H100: 989 TFLOPS ÷
3.35 TB/s ≈ **295 operations per byte**. Fetch a byte, do fewer than 295
operations with it, and you've wasted the trip.

> **Which FLOPS number?** Spec sheets list several, differing by up to 8×, and
> picking the wrong one silently corrupts every prediction you make.
>
> A GPU has two kinds of arithmetic units. **Shaders** are the
> general-purpose cores that run ordinary code; **tensor cores** are
> specialized circuits built to multiply grids of numbers, which is most of
> what machine learning does. TechPowerUp lists the RTX 3090 at "FP16 (half)
> 35.58 TFLOPS **(1:1)**", which is the *shader* rate: FP16 (half) is the
> 16-bit number format, and on shaders it runs no faster than fp32, the
> 32-bit format. Matmuls (grid-of-numbers multiplications) don't use shaders;
> they use **tensor cores**, and the 3090's dense fp16 tensor rate is
> ~71 TFLOPS. That's the number in `roofline.py`.
>
> Vendors also quote **sparse** rates, which are 2× dense and require 2:4
> structured sparsity: a model whose weights genuinely have half their
> entries as zero, a pattern you almost certainly don't have. If a headline
> number looks suspiciously round and large, check whether it's sparse.
>
> Rule: **dense tensor-core rate, at the dtype you actually run** (dtype is
> short for "number format": fp32, fp16, bf16, int8, each a different way of
> packing a number into bits).

One more wrinkle worth knowing before you trust a spec sheet: the rate is not
linear in bit width. A multiplier that handles `p`-bit inputs builds `p×p`
partial products, so halving the width from 8 to 4 bits should cut the area by
**4×**, not 2× (Nvidia's own B300 specs list FP4 at 3× the FP8 rate, and the
gap from 4× down to 3× is the fixed float-exponent circuitry plus how much die
area they *chose* to give each format). FP8 needs hardware a 3090 doesn't have
(Lecture 19); the rule here is just: **compute `F/BW` at your own dtype**, and
don't assume a "half the bits" format is "half the speed".

Now the same measure for an *algorithm*, called **arithmetic intensity**:

```
intensity = total compute (ops) / total memory traffic (bytes)
```

Compare the two and you have your answer:

- intensity **>** ops:byte → **compute-bound** (arithmetic is the limit)
- intensity **<** ops:byte → **memory-bound** (bandwidth is the limit)

### The roofline

Plot those two facts together and you get the chart the lecture is named after.
It's called a roofline because of its shape, a diagonal rising into a flat,
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

- **The diagonal**: you cannot compute faster than memory can feed you. Its
  slope *is* the memory bandwidth.
- **The horizontal**: you cannot exceed the arithmetic units. Its height *is*
  peak FLOPS.

The corner where they meet is the **ridge point**, and it sits at exactly the
ops:byte ratio (295 for an H100, 76 for a 3090; both are the same division):

```
H100   989 TFLOPS  / 3.35 TB/s  =  295.2 ops:byte
3090    71 TFLOPS  / 936.2 GB/s  =   75.8 ops:byte
```

That's the whole reason the ratio matters: it's the intensity at which the two
ceilings cross.

Now locate your algorithm on the x-axis by its arithmetic intensity:

- **Left of the ridge → memory-bound.** You're on the diagonal. A faster GPU
  changes nothing; more bandwidth is the only thing that helps.
- **Right of the ridge → compute-bound.** You're on the flat. More bandwidth
  changes nothing; more FLOPS is the only thing that helps.

That's the entire tool. One number (your intensity), one comparison (against the
ridge), and you know which half of the hardware you're wasting.

**Why this book is organized around it:** decode lands at **0.79 ops:byte**
against a ridge of 295, roughly 400× to the left, using about a quarter of one
percent of the GPU's arithmetic:

```
295 / 0.79  =  373× to the left of the ridge
0.79 / 295  =  0.27% of the arithmetic used
```

Prefill lands at ~510, on the other side. Same weights, same kernels, opposite
ceilings. (Where 510 comes from: prefill loads the weights once and does
`2 × params × N` FLOPs with them, so intensity ≈ `2·params·N / weight_bytes`,
derived in full in the "See it" section below.)

Every technique in Parts II and III is an answer to the same question: *how do I
get more arithmetic out of bytes I was going to load anyway?* Batching, KV
caching, quantization, and speculative decoding are four different answers, and
the roofline is what tells you they're all the same idea.

!!! tip "You are not meant to memorize any of this"
    Do the arithmetic once, here, to build the intuition, then let
    `book/code/roofline.py` do it forever after. Practitioners don't carry these
    figures around.

    The three sentences worth retaining: **prefill is compute-bound; decode is
    memory-bound; which one you're in depends on how many tokens share one weight
    load.** Everything else is re-derivable in ten seconds.

    [Q&A: what is the ops:byte math actually for?](qa.md#what-is-the-opsbyte-math-actually-for)

### Doing it for attention

Take unoptimized attention (`S = QK^T`, `P = softmax(S)`, `O = PV`) with
sequence length `N`, head dim `d`, FP16 (2 bytes/value), per head, batch 1.
Three matrices get materialized in memory. The jargon here is small, glossed
once: a **tensor** is a grid of numbers (this book mostly means the same thing
when it says matrix; the word also covers vectors and higher-dimensional
grids). **softmax** is the step that turns a list of scores into a probability
distribution: every score becomes a positive number and they all add up to 1.
`QK^T` and `PV` are grid multiplications (**matmuls**).

| tensor | shape | values | bytes (fp16) |
|---|---|---|---|
| Q, K, V | N × d | Nd each | 2Nd each |
| S, P | N × N | N² each | 2N² each |

Each step reads its inputs from memory, computes, and writes its output back.
Add the traffic up step by step.

**Step 1: `S = QK^T`.** Reads the two small matrices, writes the big one:

```
read  Q     Nd values    ->  2Nd bytes
read  K     Nd values    ->  2Nd bytes
write S     N² values    ->  2N² bytes
---------------------------------------------
step 1 total                  4Nd + 2N²  bytes
```

**Step 2: `P = softmax(S)`.** The score matrix must come back from memory,
get softmaxed element-wise, and the result written out again:

```
read  S     N² values    ->  2N² bytes
write P     N² values    ->  2N² bytes
---------------------------------------------
step 2 total                   4N²  bytes
```

**Step 3: `O = PV`.** P comes back a second time; V and O are the small ones:

```
read  P     N² values    ->  2N² bytes
read  V     Nd values    ->  2Nd bytes
write O     Nd values    ->  2Nd bytes
---------------------------------------------
step 3 total                  2N² + 4Nd  bytes
```

Sum the three steps:

```
memory  = (4Nd + 2N²) + (4N²) + (2N² + 4Nd)
        = 8N² + 8Nd   bytes
```

**Compute.** Two matmuls: `S = QK^T` multiplies (N,d) × (d,N) and
`O = PV` multiplies (N,N) × (N,d), each a multiply-accumulate per output
element: `2N²d` ops. Softmax is a per-element pass over the N² matrix
(subtract the row max, take exp, divide by the row sum), counted as 3 ops per
element (the max and sum reductions are cheaper passes, folded into those):

```
S = QK^T     2N²d  ops
softmax(S)   3N²   ops
O = PV       2N²d  ops
---------------------------------------
compute = 4N²d + 3N²  ops
```

Intensity, with a quick factorisation so the N-scaling is visible:

```
          compute      4N²d + 3N²      N²(4d + 3)      N(4d + 3)
intensity = -------  =  ----------  =  ----------  =  ----------
          memory        8N² + 8Nd      8N(N + d)       8(N + d)
```

Now plug in N=4096, d=128, term by term:

```
memory:
  8N² = 8 · 16,777,216         = 134,217,728 bytes   (score-matrix round-trips)
  8Nd = 8 · 4096 · 128         =   4,194,304 bytes   (Q, K, V, O themselves)
                                 ---------------
                              = 138,412,032 bytes   (~132 MiB)

compute:
  4N²d = 4 · 16,777,216 · 128  = 8,589,934,592 ops
  3N²  = 3 · 16,777,216        =    50,331,648 ops
                                 ---------------
                              = 8,640,266,240 ops

intensity = 8.64G ops / 138.4M bytes = 62.4 ops:byte
```

Against an H100's 295, attention is memory-bound by nearly 5×:

```
295 / 62.4  =  4.73× below the ridge
```

**Read that memory breakdown before moving on; it is the whole lesson.** The
useful data (Q, K, V, O) is `8Nd` = 4 MiB. The score matrices are `8N²` =
128 MiB, **32× more, and every byte of it is written to memory only to be
read straight back**: S is stored so softmax can re-read it, P is stored so the
output can re-read it. S alone is 32 MiB at this size, round-tripped for
nothing. Deleting that round-trip is exactly what FlashAttention does
(Lecture 17): tile S and P so they live in on-chip SRAM and never touch memory.

One more thing the formula gives for free. As N grows, the `Nd` terms vanish
and the intensity approaches a ceiling. Take the limit step by step: divide
numerator and denominator by N:

```
            N(4d + 3)          N(4d + 3)          (4d + 3)        4d + 3
intensity = ---------  =  ------------------  =  ----------  ->  -------
            8(N + d)        8N(1 + d/N)          8(1 + d/N)         8
                                              (d/N -> 0 as N -> ∞)
```

That is `(4d + 3)/8 = d/2 + 3/8`. For d = 128:

```
64 + 0.375  =  64.4 ops:byte
```

Attention intensity rises with N but can never exceed d/2 + 3/8; for d=128
that's 64.4, still 4.6× below the H100's ridge of 295:

```
295 / 64.4  =  4.58× below the ridge, forever
```

**Attention is memory-bound at every sequence length, on every device in the
table.** (That's check-yourself Q2, answered in advance.)

---

## See it

```bash
uv run python book/code/roofline.py
```

Confirm the book's worked example first:

```
intensity     62.4 ops:byte   (book says ~62)
```

Then the number that matters, and where it comes from so it's never a bare
figure. Decode at sequence length 2048: one step moves all the weights plus the
2048 tokens of KV already stored, and does one multiply-accumulate per weight:

```
compute  =  2 × 440.4M params                = 880.8 MFLOP
memory   =  880.8 MB weights  +  114,688 B/token × 2048 tokens
           = 880.8 MB + 234.9 MB             = 1115.7 MB
intensity = 880.8 / 1115.7                   = 0.79 ops:byte
```

(114,688 B/token = 2 × 28 layers × 8 KV heads × 128 head_dim × 2 bytes, the KV
cache size from Lecture 05.) At 512 prompt tokens prefill amortizes that same
weight load over all of them:

```
compute  =  2 × 440.4M × 512   = 451.0 GFLOP
memory   =  880.8 MB + 2 × 512 × 1024 × 2 B   (activations, one pass)
         =  880.8 MB + 2.1 MB                 = 882.9 MB
intensity = 451.0e9 / 882.9e6                 = 510.5 ops:byte
```

```
decode  (1 token, 2048 ctx)   0.79 ops:byte
prefill (512 tokens)        510.78 ops:byte
```

Two notes so the numbers stay honest. **0.79 is instantaneous at 2048 context**:
it's the intensity of *one* decode step. Lecture 01's 0.92 was the whole
generation averaged, including the KV cache growing from 512 to 768 tokens;
same algorithm, different windows. And both sit far below the ridge either way.
Second, **the ridge table**, same division for every card:

```
A100   312 TFLOPS  / 2.039 TB/s  =  153.0
3090    71 TFLOPS  / 936.2 GB/s  =   75.8
M1       2.6 TFLOPS / 68.25 GB/s =   38.1
```

**0.79 against a ridge of 295.** Decode uses roughly a quarter of one percent of
the arithmetic the machine can do. It is not a little memory-bound; it is almost
entirely memory-bound.

And the verdict table shows this holds on *every* device, H100 (ridge 295), A100
(153), 3090 (76), M1 (38). A conclusion that survives an 8× range of ridge points
(295/38 = 7.8×) is a property of the algorithm, not a quirk of one GPU.

One aside worth noticing: the **3090's ridge is 76, the H100's is 295.** The
cheaper card has proportionally *more* bandwidth per FLOP, making it relatively
better at memory-bound decode. "Slower GPU" and "worse at decode" are not the
same claim.

---

## Build it

1. Run the demo. Check 62.4 against Kiely Fig 2.18 (p.66).
2. Run `uv run pytest tests/test_02_roofline.py -v`, these pass today. Read
   them; they encode the claims above as assertions.
3. **Fill in your own hardware.** The `Device` entries are nominal spec-sheet
   figures. Find your laptop's real numbers and add them.
4. **The exercise that matters, KV cache sizing.** Kiely §5.4 (Fig 5.11, p.142)
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

- **Kiely §2.4–2.4.2** (p.61–66): the derivation this lecture reproduces.
- **Kiely §2.5** (p.67–70), how FlashAttention and PagedAttention each attack
  the numbers you just computed.
- **[Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785)**
  (Williams et al., 2009): the original. Predates GPUs in this role and still
  the clearest statement of the idea.
- **[FlashAttention](https://arxiv.org/abs/2205.14135)** (Dao et al., 2022),   §2 has the memory-traffic analysis. Skim now, implement in Lecture 17.

---

## Check yourself

Answer from your own output:

1. Decode is 0.79 ops:byte and the H100's ridge is 295. If you doubled that GPU's
   FLOPS, how much faster does decode get?
2. Attention intensity rises with N (32 at N=128, 62 at N=4096) but never passes
   the ridge. What does that tell you about attention at *any* sequence length?
3. Your predicted batch-size ceiling from the sizing exercise, what runs out
   first, and what would you change to raise it? *(Lectures 09 and 19.)*

---

## Next

**[03. Naive generation](03-naive-generation.md)**: stop predicting, start
measuring. **This is where you first write code.**

```bash
uv run python book/code/recomputation.py    # the 99.6% waste figure
```

Then: read L03, write your prediction in `notes/`, implement
`engine/model.py::load` and `generate_naive` (~30 lines), and run
`pytest tests/test_03_generation.py`.

Budget an hour. Everything before this was setup.

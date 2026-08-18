# 19. Quantization

**Build:** `kernels/triton/quant_matmul.py`, `kernels/quality_eval.py`
**Test:** `tests/test_19_quantization.py` (cuda) · **Moves:** memory, throughput, **and quality**
**Prereq:** [18. A paged attention kernel](18-paged-attention-kernel.md)

---

## The problem

Everything so far has moved bytes more cleverly. Quantization asks a blunter
question: **why are the bytes so big?**

Every weight is currently stored as a 16-bit float (FP16, 2 bytes). Decode is
memory-bound (Lecture 02, 0.79 ops:byte — 0.79 operations for every byte it
loads), which means the GPU is mostly waiting
for bytes to arrive, not computing. Time is proportional to bytes loaded, for
a simple reason: on a memory-bound kernel, the bytes *are* the work. So halve
the bytes and you nearly halve the time, no algorithmic cleverness required.
The arithmetic is the same either way; only the size of the luggage changes.

FP16 → INT8 is 2× fewer bytes (2 bytes per weight becomes 1). INT4 is 4×
(half a byte). That's a bigger decode win than anything in Lectures 16–18,
which all saved bytes too, but only in the narrow attention stage; quantization
saves bytes on the *weights*, which are the biggest single traffic item in
every decode step (880.8 MB of the 1,115.7 MB from Lecture 02).

There is a catch, and it's the whole reason this lecture is structured the way it
is: **quantization is the only optimization in this book that can make your model
worse.** Every other technique is exact, same tokens, less time. This one trades
quality for speed, and the trade is invisible unless you go looking.

---

## The idea

Store weights in fewer bits, **dequantize** on the fly: convert each small
integer back to the float it stands for, right as the kernel loads it, so the
matmul itself runs unchanged. The conversion is one multiply and one subtract:

```
w_fp16 ≈ scale × (w_int8 - zero_point)
```

Two names worth saying plainly. **scale** is how much one unit of the integer
range is worth, one step on the number line. **zero_point** is an offset: a
shift so that an integer 0 can stand for a non-zero float, and so the float 0
can be represented exactly, which matters because padding and masking sprinkle
zeros through the model. Store the tiny integer matrix and a few numbers per
group, reconstruct each weight when it is loaded, do the math as if nothing
happened.

### Where scale and zero_point come from

The formula above is the *decode* direction. You also need the *encode*
direction, and it is the thing you will actually implement, so derive it.

You have a block of real weights spanning some range `[w_min, w_max]`, and you
have exactly 256 integer slots to represent them (`int8`, values 0..255 in the
unsigned convention). Two requirements fix the mapping completely:

1. `w_min` must land on integer 0.
2. `w_max` must land on integer 255.

The float range spans `w_max − w_min`; the integer range spans `255 − 0 = 255`
steps. So one integer step is worth:

```
scale  =  (w_max − w_min) / 255
```

And the offset that puts `w_min` at integer 0 — set `w = w_min` in
`w ≈ scale × (q − zero_point)` and solve for `q`:

```
w_min = scale × (0 − zero_point)      we want q = 0 here
    ⟹  zero_point = −w_min / scale
```

rounded to an integer, since `zero_point` is stored as one.

**Work it on real numbers.** Take a block of weights with
`w_min = −0.31`, `w_max = 0.44`:

```
scale       = (0.44 − (−0.31)) / 255  =  0.75 / 255  =  0.00294

zero_point  = −(−0.31) / 0.00294      =  0.31 / 0.00294  =  105.4  →  105
```

Now quantize one weight, `w = 0.137`:

```
q  =  round(w / scale) + zero_point
   =  round(0.137 / 0.002941) + 105
   =  round(46.58) + 105
   =  47 + 105
   =  152
```

Note the `round`: `46.58` is not an integer, and that rounding is where the
information is lost. Dequantize to see exactly how much:

```
ŵ  =  scale × (q − zero_point)
   =  0.002941 × (152 − 105)
   =  0.002941 × 47
   =  0.138235

error  =  |0.137 − 0.138235|  =  0.001235      (0.90% of the value)
```

Contrast a weight that happens to land cleanly, `w = 0.10`:

```
q  =  round(0.10 / 0.002941) + 105  =  round(34.0) + 105  =  139
ŵ  =  0.002941 × 34  =  0.100000        error = 0  (exactly on a grid point)
```

So the error is not uniform — it depends on where each weight falls between two
grid points — but it is **bounded**, always, by half a step:

```
max error  =  scale / 2  =  0.002941 / 2  =  0.00147
```

That bound is the entire cost of the technique. Everything else in this lecture
is about **making `scale` smaller**, because a smaller step means a smaller
bound — and the way you shrink it is to stop sharing one scale across weights
that don't belong together.

??? question "Doesn't dequantizing on every load cost more time than it saves?"
    No, and the reason is Lecture 16's. The multiply-and-subtract happens in
    registers, on data that has *already arrived* on-chip — it costs arithmetic,
    and decode has arithmetic to spare (Lecture 02: 0.79 ops:byte against a ridge
    of 76). What you saved is the HBM traffic, which is the thing you were
    actually waiting on. You are spending the abundant resource to conserve the
    scarce one, which is the same trade FlashAttention makes.
    [Full answer](qa.md#doesnt-dequantizing-on-every-load-cost-more-time-than-it-saves)

The interesting question is what `scale` covers.

Here are the three, drawn on the same weight matrix. `s` marks one stored scale:

```
  PER-TENSOR              PER-CHANNEL             PER-GROUP
  one s for everything    one s per row           one s per 64-128 weights

  ┌──────────────┐ s      ┌──────────────┐        ┌──────┬──────┐
  │ ▪ ▪ ▪ ▪ ▪ ▪  │        │ ▪ ▪ ▪ ▪ ▪ ▪  │ s₀     │ ▪ ▪ ▪│▪ ▪ ▪ │ s₀ s₁
  │ ▪ ▪ ▪ ▪ ▪ ▪  │        │ ▪ ▪ ▪ ▪ ▪ ▪  │ s₁     ├──────┼──────┤
  │ ▪ ▪ ▪ ▪ ▪ ▪  │        │ ▪ ▪ ▪ ▪ ▪ ▪  │ s₂     │ ▪ ▪ ▪│▪ ▪ ▪ │ s₂ s₃
  │ ▪ ▪ ▪ ▪ ▪ ▪  │        │ ▪ ▪ ▪ ▪ ▪ ▪  │ s₃     ├──────┼──────┤
  └──────────────┘        └──────────────┘        │ ▪ ▪ ▪│▪ ▪ ▪ │ s₄ s₅
                                                  └──────┴──────┘
  1 scale stored          4 scales stored         6 scales stored
  worst accuracy          standard choice         best accuracy
```

**Per-tensor**: one scale for the whole matrix. Smallest metadata, worst
accuracy — and here is why, drawn as the number line from the derivation above.
One **outlier** (a single weight far outside the typical range) forces `w_max`
wide, which forces `scale` large, which makes the grid coarse *everywhere*:

```
  without the outlier:  range [-0.3, 0.3],  scale = 0.6/255  = 0.0024
  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤        fine grid where the weights actually are
 -0.3                0.3

  with one outlier at 4.0:  range [-0.3, 4.0],  scale = 4.3/255 = 0.0169
  ├────────┼────────┼────────┼────────┼────────┼────────┼────────┤
 -0.3     0.3      1.0      1.7      2.4      3.1      4.0
  └──┬──┘
     └─ every real weight is crammed in here, 7× coarser than before
        one outlier made 99.9% of the matrix less accurate
```

**Per-channel**: one scale per output channel. Standard, and much better. A
channel, for a weight matrix, is one output neuron's row of incoming weights.
Giving each row its own number line means a row of small weights gets a fine
grid regardless of what other rows contain — the outlier now damages only
*its own row*, and the other rows are untouched.

**Per-group**: one scale per group of 64–128 weights along the row. Best
accuracy, most metadata. What INT4 methods use, because 4 bits (16 levels, not
256) cannot absorb any range waste at all.

**Count the metadata**, since "most metadata" deserves a number. At group size
128, each group stores one fp16 scale (2 B) and one int8 zero-point (1 B):

```
  3 bytes of metadata  per  128 weights
  = 3 × 8 bits / 128 weights
  = 0.19 bits per weight
```

So "INT4" is really about **4.19 bits per weight** once you count the scales —
a 4.7% overhead on the thing you were trying to shrink. Halve the group size to
64 and you double that overhead to 0.38 bits/weight for a finer grid. That
trade — accuracy against metadata — is exactly what the group size dial
controls.

### Weights vs. activations vs. KV cache

First, what those words mean, because the whole section turns on the difference.

**Weights** are the model's learned parameters — the matrices trained once and
then frozen. **Activations** are the values that flow *through* those matrices:
computed fresh for every token, used immediately, thrown away. In a single
matmul:

```
        x    ──────►  [  W  ]  ──────►    y

     activation        weight          activation
   this token only   every token,     this token only
   ~2 KB, transient   forever         ~2 KB, transient
                     880 MB, resident
```

The consequences are what matter:

```
                    weights                  activations
   when known    training time            only at runtime
   how often     loaded every step        recomputed every token
   varies with   nothing                  the input
   decode cost   880 MB/step (dominant)   small
```

Because weights are fixed you can inspect their range offline, pick scales
carefully, even repair them (GPTQ, AWQ below). Activations you meet for the
first time mid-request — you cannot study a distribution you have not seen yet,
which is why activation quantization is the harder half.

The **KV cache** is a third thing: activations you decided to *keep*. Keys and
values are computed like any activation, but instead of being discarded they are
stored for every future token — which is why they get their own row below.

With that settled, three separate decisions:

**Weight-only (W8A16, W4A16)**: quantize weights, compute in FP16. The notation
is a pair: the weight precision, then the activation precision. W8A16 means
8-bit weights, 16-bit activations; W4A16 the same with 4-bit weights. Dominant
for inference. Decode is bound by *weight* traffic, so this attacks the bottleneck
directly, and activations stay accurate. Dequantization happens in-kernel.

**Weight + activation (W8A8)**: both quantized, so the matmul itself runs on
INT8 tensor cores (the chip's specialist matrix-multiply hardware) instead of
dequantizing back to FP16 first. Faster in principle — and much harder, for the
reason above: activation ranges are only known at runtime, and they contain
outliers that a per-tensor scale cannot absorb. This is what SmoothQuant below
exists to fix.

**KV cache quantization**: a different axis entirely. From Lecture 05 the cache can
exceed the model's size; from Lecture 09 its capacity caps your batch size.
Compressing it buys concurrency rather than per-step speed. The
[field notes](field-notes.md) flag a recurring caveat: KV compression that looks
fine on general text can degrade **reasoning** specifically.

Quantization changes the *bytes per entry*; structural methods (MLA, sparse
attention, token merging, Lecture 05's "second lever") change the *number* of
entries. The two compose, and the field-notes caveat applies to both.

### The methods worth knowing

The objective here is being able to choose: when round-to-nearest isn't good
enough, each method below is a different way of making rounding hurt less,
distinguished by what it spends its effort on.

| Method | Idea |
|---|---|
| **RTN** | round-to-nearest; the baseline, no calibration (no pass over sample data to set the scales) |
| **GPTQ** | layer-wise, compensating for error (the mismatch each rounded weight introduces) using second-order information |
| **AWQ** | protect the ~1% of channels that matter most, based on activation magnitude |
| **SmoothQuant** | shift outlier difficulty from activations into weights |

AWQ's premise is worth stating because it generalizes: **not all weights matter
equally.** A small fraction of channels carry disproportionate influence — the
activations that flow through them are consistently the largest, so rounding
errors there get multiplied by the biggest values — and keeping those at
higher precision recovers most of the quality.

Two table entries read as magic; unpack them. GPTQ's "second-order information"
is the shape of the error surface, measured by running a few batches of data
through the layer (a **calibration** pass). It lets GPTQ round each weight in a
way that compensates for the damage the previous rounding just did, rather than
rounding every weight in isolation. SmoothQuant's "shift" means multiplying the
outlier-prone activations down and the weights up, paired per channel, so the
hard quantization job moves from the activations (where it cannot be done
well) into the weights (where it can be measured and compensated). Both are
efforts to make rounding hurt less, not new math.

### Hardware decides your format

The [field notes](field-notes.md) record an operator choosing a quant specifically
because **3090s accelerate INT4 in hardware**. FP8 needs Ada/Hopper or newer; a
3090 (Ampere) doesn't have it.

Two caveats on that, because "accelerates INT4" is easy to over-read. Ampere's
INT4 tensor-core path is real but was **deprecated in later architectures**, so
it is a reason specific to that generation rather than a durable one. And the
mainstream INT4 methods you are most likely to run — GPTQ and AWQ in the W4A16
setting — do **not** use it: they store 4-bit weights and *dequantize to FP16*
in the kernel, exactly as the opening of this lecture describes. Their win is
the halved (then quartered) weight traffic, not an INT4 matmul.

The operator's point survives both caveats, and it is the durable one: the
format that is fastest for you is a property of the silicon in front of you.

The same operator kept **linear attention layers at full precision** while
quantizing the rest, because those layers quantize poorly. "Quantize the model"
is rarely the actual operation; mixed precision across layer types is normal.

**So: your format is chosen by your silicon, not by a leaderboard.**

---

## Measuring what it costs

The objective here is quantifying the quality trade before you trust any
speedup — the speed is easy to measure, the damage is not. This is the part
people skip, and it's the reason this lecture exists.

**Perplexity is necessary and not sufficient.** Perplexity is one number: how
surprised, on average, the model is by each next token in a corpus, lower is
better. It's an average over next-token
predictions; a small delta can hide a specific broken capability. A model can hold
perplexity nearly constant while losing the ability to follow a multi-step format.

The [field notes](field-notes.md) describe a much better methodology, from a
community comparison across BF16 (bfloat16, a 16-bit float) → Q8 → Q6 → Q5 →
Q4 → IQ3 (a 3-bit variant). Two principles:

**Grade a task with a verifiable answer.** Theirs: given a chess PGN, track the
board state and render it as SVG. Degradation appeared as *wrong piece placement*
and *wrong board orientation*, structured-reasoning failures a perplexity number
would never surface.

**Use out-of-distribution inputs.** They chose deliberately terrible chess moves
"no player above 300 elo would ever play," so the model couldn't substitute
memorization for reasoning. If your eval is in the training data, you're measuring
recall.

Your quality harness should have both properties. Anything less and you're
shipping a regression you can't see.

---

## Build it

1. Implement per-channel INT8 weight quantization and a Triton dequant-matmul in
   `kernels/triton/quant_matmul.py`.
2. `uv run pytest tests/test_19_quantization.py -v`, numerics within tolerance
   against FP16.
3. **Build `kernels/quality_eval.py` before you benchmark speed.** A task with a
   gradeable answer, on inputs the model can't have memorized. Format-following
   and multi-step arithmetic are good cheap choices.
4. Measure **all three axes** at FP16 / INT8 / INT4:

| | memory | tok/s | quality |
|---|---|---|---|
| FP16 | baseline | baseline | baseline |
| INT8 | | | |
| INT4 | | | |

5. Try KV cache quantization separately, and measure the *concurrency* gain.

**Predict first:** decode is memory-bound, so what speedup should INT8 give? Does
your measurement match, and if not, what else is now the bottleneck?

---

## What you should see

**Memory roughly halves at INT8.** Predictable.

**Speedup below 2×, and you can predict the number before you measure it.** You
quantized weights, not everything — activations, KV cache, and overhead are
unchanged. Lecture 02 counted what a decode step actually moves:

```
weights                     880.8 MB      ← the only part INT8 shrinks
KV cache + activations      234.9 MB      ← unchanged
                          ──────────
total per step            1,115.7 MB
```

Weights are `880.8 / 1,115.7 = 78.9%` of the traffic. Halve *that* term and
leave the rest alone:

```
INT8 step  =  880.8/2  +  234.9   =  440.4 + 234.9  =  675.3 MB

speedup    =  1,115.7 / 675.3     =  1.65×
```

So **~1.65×**, not 2×, and now you know the number to check your measurement
against. This is Amdahl's law in miniature: the speedup of a change is capped by
the share of the time the change actually touches. Weights are the biggest
share, so you get most of the win — but the 21% you didn't touch takes exactly
as long as it always did.

Work the same arithmetic for INT4 before you run it. Weights go to a quarter:

```
INT4 step  =  880.8/4  +  234.9   =  220.2 + 234.9  =  455.1 MB
speedup    =  1,115.7 / 455.1     =  2.45×
```

Note what happened: halving the weights *again* bought only another 0.8×, not
another 1.65×. The unquantized 234.9 MB is now more than half the step, and it
is the new ceiling. Chasing bits has diminishing returns for a reason you can
compute in advance, and that is precisely when attention should shift to the KV
cache instead.

**INT8 quality nearly indistinguishable** on most tasks. This is why W8A16 is
close to a default.

**INT4 sometimes visibly worse**, and *task-dependently* so. Your eval decides
whether it's acceptable, which is precisely why you built it first.

---

## Go deeper

- **[AWQ](https://arxiv.org/abs/2306.00978)** (Lin et al.): the salient-weight
  argument; §3 is the core.
- **[GPTQ](https://arxiv.org/abs/2210.17323)** (Frantar et al.), layer-wise
  second-order quantization.
- **[SmoothQuant](https://arxiv.org/abs/2211.10438)** (Xiao et al.), moving
  outlier difficulty from activations to weights.
- **Kiely §5.1–5.1.3** (p.120–128), number formats, approaches, and §5.1.3
  specifically on measuring quality impact.
- **[Field notes](field-notes.md)**: hardware-specific format choice, mixed
  precision across layer types, and the task-based eval methodology above.

---

## Check yourself

1. Why does quantization help decode more than prefill?
2. INT8 halves weight bytes but gives ~1.6× speedup. Where did the rest go?
3. Perplexity moved 0.3%. Why is that not sufficient evidence of safety?
4. Why did that operator keep linear attention layers at full precision?
5. Same model, INT4: acceptable for a chat product, not for a coding agent. Give a
   concrete reason grounded in your own eval.

---

## Next

**[20. Raw CUDA](20-raw-cuda.md)**: one level lower, to see what Triton was
doing on your behalf.

**You will probably not beat Triton, and that's the point.** Pick one kernel,
write it by hand, then use `ncu` to find out exactly where Triton wins. The
durable outcome is being able to read vLLM's CUDA, not a faster kernel.

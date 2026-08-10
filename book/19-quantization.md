# 19 — Quantization

**Build:** `kernels/triton/quant_matmul.py`, `kernels/quality_eval.py`
**Test:** `tests/test_19_quantization.py` (cuda) · **Moves:** memory, throughput, **and quality**
**Prereq:** [18 — A paged attention kernel](18-paged-attention-kernel.md)

---

## The problem

Everything so far has moved bytes more cleverly. Quantization asks a blunter
question: **why are the bytes so big?**

Decode is memory-bound (Lecture 02, 0.75 ops:byte). Time is proportional to bytes
loaded. So halve the bytes and you nearly halve the time, no algorithmic
cleverness required.

FP16 → INT8 is 2× fewer bytes. INT4 is 4×. That's a bigger decode win than
anything in Lectures 16–18.

There is a catch, and it's the whole reason this lecture is structured the way it
is: **quantization is the only optimization in this book that can make your model
worse.** Every other technique is exact, same tokens, less time. This one trades
quality for speed, and the trade is invisible unless you go looking.

---

## The idea

Store weights in fewer bits, dequantize on the fly:

```
w_fp16 ≈ scale × (w_int8 - zero_point)
```

The interesting question is what `scale` covers.

**Per-tensor** — one scale for the whole matrix. Smallest metadata, worst accuracy:
a single outlier stretches the range and crushes precision for everything else.

**Per-channel** — one scale per output channel. Standard, and much better.

**Per-group** — one scale per group of 64–128 weights. Best accuracy, most
metadata. What INT4 methods use, because 4 bits can't absorb any range waste.

### Weights vs. activations vs. KV cache

Three separate decisions, often confused:

**Weight-only (W8A16, W4A16)** — quantize weights, compute in FP16. Dominant for
inference. Decode is bound by *weight* traffic, so this attacks the bottleneck
directly, and activations stay accurate. Dequantization happens in-kernel.

**Weight + activation (W8A8)** — both quantized, so the matmul runs on INT8 tensor
cores. Faster in principle, but activations have outliers that make them much
harder to quantize than weights.

**KV cache quantization** — a different axis entirely. From Lecture 05 the cache can
exceed the model's size; from Lecture 09 its capacity caps your batch size.
Compressing it buys concurrency rather than per-step speed. The
[field notes](field-notes.md) flag a recurring caveat: KV compression that looks
fine on general text can degrade **reasoning** specifically.

### The methods worth knowing

| Method | Idea |
|---|---|
| **RTN** | round-to-nearest; the baseline, no calibration |
| **GPTQ** | layer-wise, compensating for error using second-order information |
| **AWQ** | protect the ~1% of channels that matter most, based on activation magnitude |
| **SmoothQuant** | shift outlier difficulty from activations into weights |

AWQ's premise is worth stating because it generalizes: **not all weights matter
equally.** A small fraction of channels carry disproportionate influence, and
keeping those at higher precision recovers most of the quality.

### Hardware decides your format

The [field notes](field-notes.md) record an operator choosing a quant specifically
because **3090s accelerate INT4 in hardware**. FP8 needs Ada/Hopper or newer; a
3090 (Ampere) doesn't have it.

The same operator kept **linear attention layers at full precision** while
quantizing the rest, because those layers quantize poorly. "Quantize the model"
is rarely the actual operation; mixed precision across layer types is normal.

**So: your format is chosen by your silicon, not by a leaderboard.**

---

## Measuring what it costs

This is the part people skip, and it's the reason this lecture exists.

**Perplexity is necessary and not sufficient.** It's an average over next-token
predictions; a small delta can hide a specific broken capability. A model can hold
perplexity nearly constant while losing the ability to follow a multi-step format.

The [field notes](field-notes.md) describe a much better methodology, from a
community comparison across BF16 → Q8 → Q6 → Q5 → Q4 → IQ3. Two principles:

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

**Speedup below 2×.** You quantized weights, not everything, activations, KV
cache, and overhead are unchanged. Amdahl again.

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
- **[Field notes](field-notes.md)** — hardware-specific format choice, mixed
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

**[20 — Raw CUDA](20-raw-cuda.md)** — one level lower, to see what Triton was
doing on your behalf.

**You will probably not beat Triton, and that's the point.** Pick one kernel,
write it by hand, then use `ncu` to find out exactly where Triton wins. The
durable outcome is being able to read vLLM's CUDA, not a faster kernel.

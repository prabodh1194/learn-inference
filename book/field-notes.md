# Field notes

What practitioners report, gathered from r/LocalLLaMA. Books give you the
mechanism; these give you the magnitudes and the disappointments.

Treat them as **calibration, not gospel** — hardware, model, and quant differ,
and a single reported number is not a benchmark. Where a claim here contradicts
what you measure yourself, believe your own measurement.

Each entry is tagged with the lecture it informs.

---

## The prefill/decode gap is enormous in practice — **L01**

A 2×3090 setup running Qwen3.5-27B dense
([source](https://old.reddit.com/r/LocalLLaMA/comments/1rianwb/running_qwen35_27b_dense_with_170k_context_at/))
reports:

- **prefill ~1500 tok/s**
- **decode ~100 tok/s** (rarely below 60)

A **15× gap on the same hardware and model.** This is Lecture 01's asymmetry as a
production number, and it's why TTFT and TPS get tuned separately.

## Batching multiplies aggregate throughput — **L07, L08**

Same setup: **~100 tok/s for one user, 585 tok/s across 8 concurrent requests.**

Per-user speed barely moved; aggregate went up ~6×. Exactly the prediction from
Lecture 01's batching table — memory traffic is fixed, so extra sequences ride
along nearly free.

It also shows why "tokens per second" is a near-meaningless claim without saying
**per user or aggregate**. Those two numbers differ by 6× here, and by far more
at larger batch sizes.

## Speculative decoding: tune by acceptance rate, not by docs — **L12**

Two findings that matter more than any tutorial:

**Documented defaults are a starting point, not an answer.** The 2×3090 operator
found docs recommending 3 draft tokens; measuring *mean acceptance length* showed
5 was better for their workload — and above 5, performance got measurably
**worse**. More speculation is not more speed: rejected drafts cost real compute.

**The workload decides the win.** A widely-upvoted report on Gemma-class models
with a small draft model: **+29% average, +50% on code.** Code is repetitive and
predictable, so drafts get accepted; prose is novel, so they don't.

This is precisely why `bench/workloads.py` ships both `code_completion` and
`prose`. If you only benchmark one, you'll conclude speculative decoding is
either magic or useless — and you'll be wrong either way.

## Quantization interacts with your specific hardware — **L19**

The 3090 operator picked a quant specifically because **3090s have hardware INT4
support**, and separately noted that **linear attention layers quantize poorly**,
so they kept those at full precision while quantizing the rest to int4.

Two lessons:
1. The right quantization format depends on what your silicon accelerates — this
   is not a pure accuracy/size tradeoff.
2. **Mixed precision across layer types** is normal in practice. "Quantize the
   model" is rarely the actual operation.

## Tensor parallelism benefits from interconnect — **L22**

The same operator credits **NVLink** for part of their TP performance. TP
all-reduces after every layer, so interconnect bandwidth is directly on the
critical path.

Relevant to your rentals: two GPUs in one box without NVLink will scale worse
than the numbers you read in blog posts. Measure your own scaling curve (L22)
rather than assuming linear.

## Building from source is usually not the win — **L26**

Same operator compiled vLLM from scratch and reported it "doesn't seem to
increase the performance much."

A useful prior: **configuration and workload shape dominate build flags.** Before
reaching for exotic builds, tune engine arguments and check you're measuring the
right workload. The boring lever is usually the bigger one.

---

## How to use this

When a lecture predicts something, check whether the field notes agree on
*magnitude*. If your measurement disagrees with both, you have a bug. If it
disagrees only with the field notes, you may have found something interesting —
different hardware, different model, different workload.

**Add your own entries.** When you measure something that surprises you, write it
here with the date and your setup. That's how this file stays worth reading.

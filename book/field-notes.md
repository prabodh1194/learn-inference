# Field notes

What practitioners report, gathered from r/LocalLLaMA. Books give you the
mechanism; these give you the magnitudes and the disappointments.

Treat them as **calibration, not gospel**, hardware, model, and quant differ,
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
Lecture 01's batching table, memory traffic is fixed, so extra sequences ride
along nearly free.

It also shows why "tokens per second" is a near-meaningless claim without saying
**per user or aggregate**. Those two numbers differ by 6× here, and by far more
at larger batch sizes.

## Speculative decoding: tune by acceptance rate, not by docs — **L12**

Two findings that matter more than any tutorial:

**Documented defaults are a starting point, not an answer.** The 2×3090 operator
found docs recommending 3 draft tokens; measuring *mean acceptance length* showed
5 was better for their workload, and above 5, performance got measurably
**worse**. More speculation is not more speed: rejected drafts cost real compute.

**The workload decides the win.** A widely-upvoted report on Gemma-class models
with a small draft model: **+29% average, +50% on code.** Code is repetitive and
predictable, so drafts get accepted; prose is novel, so they don't.

This is precisely why `bench/workloads.py` ships both `code_completion` and
`prose`. If you only benchmark one, you'll conclude speculative decoding is
either magic or useless, and you'll be wrong either way.

## Quantization interacts with your specific hardware — **L19**

The 3090 operator picked a quant specifically because **3090s have hardware INT4
support**, and separately noted that **linear attention layers quantize poorly**,
so they kept those at full precision while quantizing the rest to int4.

Two lessons:
1. The right quantization format depends on what your silicon accelerates, this
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

## Measure quality with a *task*, not just perplexity — **L19**

A systematic comparison of Qwen3-27B across BF16 → Q8 → Q6 → Q5 → Q4 → IQ3
([source](https://old.reddit.com/r/LocalLLaMA/comments/1t53dhp/quality_comparison_between_qwen_36_27b/))
didn't use perplexity. It used a task with a **verifiable right answer**: given a
chess PGN, track the board state and render it as SVG.

Two things worth stealing:

**A task exposes what perplexity hides.** Degradation showed up as *wrong piece
placement* and *wrong board orientation*, structured-reasoning failures that a
small perplexity delta wouldn't reveal.

**Deliberately out-of-distribution inputs.** The author used nonsense chess moves
"no player above 300 elo would ever play," specifically so memorization couldn't
substitute for reasoning. If your eval is in the training set, you're measuring
recall.

When Lecture 19 says "measure the quality axis," this is the bar: a task you can
grade, on inputs the model can't have memorized.

## Conventional wisdom about parallelism is worth re-measuring — **L22**

An operator with **2× GH200 and no NVLink** (PCIe only; they quote 125 GB/s
against NVLink's 900) followed the standard guidance that low interconnect bandwidth means
you should use pipeline parallelism instead of tensor parallelism
([source](https://old.reddit.com/r/LocalLLaMA/comments/1qa1guo/i_bought_a_9k_gh200_desktop_to_save_127_on_claude/)).

**Pipeline parallel lost. TP2 won**, on the exact hardware profile where the
guides say it shouldn't.

They also found `--max-num-seqs 16` was the single knob that "controls whether it
feels like a sports car or a fax machine", a scheduler concurrency limit
(Lecture 08's `max_batch_size`) mattering more than the parallelism strategy.

The lesson isn't "TP always wins." It's that **rules of thumb about interconnect
and parallelism are workload- and model-dependent**, and a weekend of
benchmarking beats a blog post. Measure your own scaling curve in Lecture 22.

## KV cache quantization is its own frontier — **L19, L05**

Reports circulate of projects compressing the **KV cache** rather than the
weights, claiming 3–5× compression. Unlike the entries above, I have no specific
source for this one, treat the number as hearsay until you measure it. Worth knowing because it attacks a different
bottleneck than weight quantization: from Lecture 05, the cache can exceed the
model's size at long context, and from Lecture 09, cache capacity directly caps
your batch size.

Note the recurring caveat in these reports: compression that holds up on general
text can degrade **reasoning** specifically. Another argument for task-based
evaluation over aggregate metrics.

---

## How to use this

When a lecture predicts something, check whether the field notes agree on
*magnitude*. If your measurement disagrees with both, you have a bug. If it
disagrees only with the field notes, you may have found something interesting —
different hardware, different model, different workload.

**Add your own entries.** When you measure something that surprises you, write it
here with the date and your setup. That's how this file stays worth reading.

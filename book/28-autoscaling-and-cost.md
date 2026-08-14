# 28. Autoscaling and cost

**Build:** `serve/autoscale.py`, `bench/cost_model.py` · **Test:** `tests/test_28_cost.py`
**Moves:** dollars per million tokens: the number that decides whether any of this matters
**Prereq:** [27. Routing and disaggregation](27-routing-and-disaggregation.md)

---

## The problem

Every optimization in this book has been measured in tokens per second. Nobody
buys tokens per second. They buy **tokens**, and someone pays for the GPU-hours.

The number that actually matters:

```
cost per million tokens = (GPU $/hour) / (tokens/hour) × 1,000,000
```

It's the metric that unifies everything you've built, and it exposes a cost that
tokens/sec completely hides.

---

## The idea

### Fleet utilization dominates

A note on the word, because it does two jobs in this book. **Fleet utilization**
(the subject of this section) is the fraction of your paid GPU-hours that do paid
work. **GPU
busy-percentage** is what `nvidia-smi` reports for one card, and it's the
misleading one (see the autoscaling signal section). They are unrelated.

Work an example. A 3090 at $0.25/hour sustaining 2,000 tok/s:

```
2000 tok/s × 3600 = 7.2M tokens/hour
$0.25 / 7.2M × 1M = $0.035 per million tokens
```

Now at 20% utilization, which is what a real service with diurnal traffic looks
like. Every number above gets multiplied by the utilization:

```
7.2M × 0.20 = 1.44M tokens/hour
$0.25 / 1.44M × 1M = $0.174 per million tokens

ratio vs full utilization:  $0.174 / $0.035  =  4.97x
```

**5× more expensive than the same engine at 100% utilization, and the engine
didn't change.** (Against a more realistic 80% baseline: 7.2M × 0.8 = 5.76M/hr,
$0.25 / 5.76M × 1M = $0.043/M, and $0.174 / $0.043 = 4.0×; 4× more.)

This is the punchline of Part V: **at low utilization, utilization dominates every
kernel optimization in Part III.** Put numbers on it: raising utilization from
20% to 50% multiplies tokens per GPU-hour by `50/20 = 2.5×`. A 30% faster
attention kernel, at roughly 20% of decode runtime (Lecture 15's table), buys
`30% × 20% = 6%` end-to-end. Knowing which lever you're pulling is the skill.

### Autoscaling and its costs

Scale replicas with demand, but scaling is not free.

**Cold starts are brutal.** Provision a GPU node, pull a multi-gigabyte container,
load weights, warm up CUDA graphs. **Minutes**, not seconds. Scale up reactively
and you're always behind the traffic.

Mitigations, roughly in order of usefulness:
- **Predictive scaling** on known daily patterns
- **Warm pools**: idle-but-loaded replicas, paying to avoid latency
- **Faster loading**: cached images, `safetensors`, streaming weights
- **Scale to zero** for dev and spiky low-volume workloads only

**Scale on the right signal.** CPU utilization is meaningless here. GPU
busy-percentage is misleading, a memory-bound decode loop shows high utilization while doing
little work. Scale on **queue depth** or **concurrent sequences**: queue depth
growing is the honest saturation signal from Lecture 25.

### The knee is your capacity number

From Lecture 25: capacity isn't throughput, it's **throughput at an acceptable
p99**. Autoscale to keep each replica *below* its knee.

Running replicas past the knee to "use them fully" is the classic mistake, you get
marginally more throughput and unbounded latency growth.

### Where the money actually goes

| Lever | Effect | Lecture |
|---|---|---|
| Fleet utilization | **largest** at low load | this one |
| Batch size | more tokens per GPU-hour | 07–09 |
| Quantization | cheaper GPU, or bigger batches | 19 |
| Prefix caching | fewer tokens computed at all | 10 |
| Spot instances | 60–80% cheaper, can vanish |, |
| Kernels | real, but smaller than the above | 15–20 |

Sobering and worth sitting with: **prefix caching can beat every kernel
optimization combined**, because not computing a token is cheaper than computing
it quickly. The same is true of a good cache-aware router.

Kernel work is the most technically satisfying and rarely the highest-leverage.
Both things are true.

---

## Build it

1. `bench/cost_model.py`, takes measured throughput and GPU price, reports cost
   per million tokens across utilization levels.
2. Compute it for **your** engine and for **vLLM** (Lecture 26's numbers), at 20%,
   50%, and 90% utilization.
3. `serve/autoscale.py`, scale on queue depth, with a configurable target and
   cooldown.
4. **Measure your cold start.** Time from "scale up" to "serving traffic." Be
   honest about it; it's usually worse than people assume.
5. Simulate a diurnal traffic pattern and compare: fixed capacity for peak, versus
   autoscaled. Report cost *and* p99.

**Predict first:** what's your cost per million tokens at 50% utilization? Compare
to the published price of a hosted API for a similar model. The gap, in either
direction, is informative.

---

## What you should see

**Cost dominated by fleet utilization**, not by engine quality, below ~50%.

**Cold starts measured in minutes.** This is why scale-to-zero is unsuitable for
latency-sensitive production.

**Autoscaling saving real money on diurnal traffic**: and hurting p99 during
scale-up events. There's the trade.

**Your cost above hosted APIs at low volume.** They amortize across many customers,
which is a genuine structural advantage. At high sustained volume the picture can
reverse.

---

## Go deeper

- **Kiely §7.2–7.2.5** (p.183–192), autoscaling, cold starts, routing, scale to
  zero, independent component scaling.
- **Kiely §7.4.2** (p.201), cost estimation.
- **Kiely §7.3** (p.193), multi-cloud capacity, GPU procurement, reliability.
- **Kiely §7.4.3** (p.203), observability: what to actually monitor.
- **[Field notes](field-notes.md)**: the €9k GH200 bought "to save $1.27 on Claude
  Code." Funny, and a real lesson about amortization at low volume.

---

## Check yourself

1. Same engine, 20% vs 80% utilization. How much does cost per million tokens
   change, and why?
2. Why is GPU busy-percentage a poor autoscaling signal for LLM serving?
3. Your cold start is 4 minutes. What does that rule out?
4. Why can prefix caching beat every kernel optimization in Part III on cost?
5. Your cost is 3× a hosted API. Name two structural reasons, and one thing you'd
   change.

---

**Part V complete.** You've built an engine, made it fast, made it a service,
measured it honestly, and priced it.

## Next

**[29. Contributing](29-contributing.md)**: go work on the real thing.

Nothing left to build here. The remaining gap between you and someone who does
this professionally is **context**: which problems are open, which tradeoffs are
contested. You get that by participating.

The one habit that makes a contribution land: **bring a measurement.**

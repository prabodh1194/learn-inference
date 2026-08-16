# 28b. Reasoning and test-time compute

**Build:** `bench/reasoning_cost.py` · **Test:** `tests/test_28b_reasoning.py`
**Moves:** dollars per *correct answer* — the number that decides how much thinking to buy
**Prereq:** [28. Autoscaling and cost](28-autoscaling-and-cost.md)

---

## The problem

Lecture 28 priced tokens. Reasoning models changed what a token is *for*: a
model can now "think" — emit thousands of chain-of-thought tokens before its
first answer token — and those thinking tokens are ordinary decode tokens. They
cost exactly what Lecture 01 says they cost. The result is that accuracy has
become a *serving-cost* dial, and someone has to turn it.

This lecture is about the turning.

---

## The idea

### Thinking is decode, priced as decode

There is no new cost model. Every chain-of-thought token is one more `weights ×
1` re-read. A reasoning model that spends 10,000 tokens thinking before it
answers has moved:

```
10,000 × 840 MiB ≈ 8 TiB  of weight traffic
```

against 840 MiB for a one-token answer. The cache grows as it thinks, too. By
the *last* token of the trace it holds

```
10,000 tokens × 112 KiB/token  =  1,120,000 KiB  ≈  1.1 GiB
```

which is larger than the 840 MiB weight read — so late in a long trace the
cache, not the weights, is the thing decode waits on.

Be careful with that 1.1 GiB: it is the cache at the *end*, not throughout. The
cache starts empty and grows one token at a time, so the step-by-step cost ramps
linearly and the *average* over the trace is about half the peak:

```
step     1:  ~0 GiB of cache      +  840 MiB of weights
step  5,000:  0.55 GiB            +  840 MiB
step 10,000:  1.10 GiB            +  840 MiB
             ────────────────────────────────
   average:  ~0.55 GiB of cache traffic per step
```

Either way the conclusion holds — a long trace drags the whole session into the
long-context regime where the cache (L05) and its compression (L19) dominate —
but the crossover happens partway through the trace, not at the first token.
"Test-time compute" is not free compute: it is tokens, and tokens are bytes.

### Accuracy scales with tokens, sub-linearly

The reason models think at all is that more tokens buy more accuracy — but the
returns diminish.

??? question "Why would writing more words make an answer *more correct*?"
    Because the extra information isn't recalled, it's **computed**. One forward
    pass has a fixed depth, which caps how much serial reasoning can happen
    before the model must commit to a token — so a problem needing more steps
    than that is unsolvable in one shot regardless of what the model knows. Each
    generated token is fed back in as input, so the model's own output becomes
    scratch memory and a chain of N tokens buys roughly N times the serial
    depth. That's why it helps enormously on multi-step arithmetic and proofs,
    and barely at all on factual recall — recall needs no serial depth.
    [Full answer](qa.md#why-would-generating-more-tokens-make-an-answer-more-correct)

Two measured curves make the shape concrete:

- **Longer traces.** During DeepSeek-R1-Zero's training, average response length
  grows from hundreds to thousands of tokens *alongside* accuracy. Nobody told
  it to think longer; the reward signal only cares about landing the answer, and
  thinking longer is what happens to work — up to the point it doesn't pay.
- **Majority voting.** R1-Zero's AIME pass@1 climbs from 15.6% to **77.9%** over
  training, and voting across 64 sampled answers lifts that to **86.7%** —
  roughly 9 points for 64× the *tokens*. Linear cost, log-ish gain. That gap is
  the whole economics of reasoning.

A caveat on that "64×", because this book has spent two parts on it: 64× the
tokens is **not** 64× the cost or 64× the time. The 64 samples share one prefill
(the question is prefilled once), and their decodes batch together — which is
precisely the regime Lecture 07 showed is nearly free, since decode is
memory-bound and the batch amortizes one weight read across all 64 sequences.
On a loaded server the marginal cost of voting is far below 64×; on an idle one
decoding a single sample at a time, it really is. **Which number you get depends
on your batch, not on the algorithm.**

Both curves have the same shape, and it is the shape that matters. Sketching
the voting numbers on a log axis — each step right is a *doubling* of samples:

```
 accuracy
   90% ┤                                            ● 86.7% (64 votes)
       │                                    ●
   80% ┤                            ●
       │  ● 77.9% (1 sample, pass@1)
   70% ┤
       └────┬───────┬───────┬───────┬───────┬───────┬────>  samples
            1       2       4       8      16      64       (log scale)

       each step right DOUBLES the tokens you spend
       each step up buys steadily LESS accuracy
```

Read the axes carefully, because the log scale is doing the work. Cost grows
along the bottom *multiplicatively* — 1, 2, 4, 8 — while accuracy climbs the
side *additively*, and by ever-smaller amounts. That is what "linear cost,
log-ish gain" means: to keep winning equal slices of accuracy you must keep
doubling the spend. The curve never turns down, which is why it is tempting;
it just flattens, which is why there is always a point where the next doubling
is not worth buying.

### The context ceiling

There is a hard bound on how much thinking a single sequence can do: `max_seq_len`.
You can spend *parallel* compute freely (sample many branches, vote, self-check),
but *sequential* tokens are capped by the context window, which is KV capacity
(L09). Techniques like **budget forcing** ("think for exactly N tokens then
stop") and self-consistency exist precisely to spend compute while staying under
that ceiling. The KV cache is what actually prices long reasoning.

### The distillation economics

The cheapest way to get a good reasoner is not to train a big one, but to
**distill** a big one's reasoning into a small one and let the small one think
longer. Decode cost scales with `params × tokens` — but *which* params, and
this is where it is easy to get the answer badly wrong.

The tempting arithmetic uses the headline size:

```
671B model ×   1 token    =  671B-param-token
 32B model ×  ~21 tokens  =  671B-param-token      671 / 32 ≈ 21
```

**That is wrong, and Lecture 23 is why.** DeepSeek-V3 is 671B *total* but only
**37B active** per token: the router picks a handful of experts and the rest of
the network is never read. Decode is memory-bound, so its per-token cost tracks
the bytes actually read — the active parameters — not the parameters sitting
resident in VRAM. Lecture 23 put it plainly: the MoE's traffic per token "is why
a 671B-parameter model can decode like a 37B one."

Redo it with active parameters, which is what decode actually pays:

```
DeepSeek-V3 :  37B active × 2 B  =  74 GB per token
32B dense   :  32B        × 2 B  =  64 GB per token

ratio  =  74 / 64  =  1.16×
```

So the honest number is **~1.16×, not 21×**. Per decoded token, a 32B dense
model and a 671B/37B-active MoE cost roughly the same. The small model does
*not* get to think 21× longer for free; it gets maybe 16% more tokens per
dollar of decode.

Which raises the obvious question: then why distill at all? The win is real but
it is not where the naive arithmetic put it:

- **Memory, not bandwidth.** The MoE must hold all 671B resident — `671B × 2 B
  = 1.34 TB`, a multi-node deployment. The 32B dense model is `64 GB`, one
  GPU. That is the difference between a cluster and a card, and it is a
  *capacity* argument, not a per-token one.
- **Batching and utilization.** One card serving many sequences (L28's fleet
  utilization) beats a cluster held for one model, and the small model reaches
  a workable batch far sooner (L28's batch floor).
- **The reasoning ability itself.** Direct RL on the small model "mostly
  stalls" — the capability is discovered by the big model and *copied* cheaply.
  That is the actual product of distillation.

This is why the models you serve are distilled reasoners thinking long, not
giants answering short — but the reason is deployment footprint, not a 21×
token discount that does not exist.

### Verifiers make it cheap to pick

When the task has a *verifiable* answer (tests pass, the math checks, the board
is legal), you don't need the model to be right on one shot — you need it to
generate candidates a verifier can rank. And from Lecture 24b: **scoring N
candidates is one compute-bound prefill; generating them is N memory-bound
decodes.** Verifiable tasks flip the economics entirely: the expensive part is
sampling, the cheap part is picking, so spend the sampling budget on diversity
and let the verifier do the judging.

---

## The code

```python
def reasoning_cost(active_params, tokens, bytes_per_param, bandwidth, usd_per_hour):
    # active_params, NOT total: decode reads only the experts the router picked.
    # For a dense model the two are the same; for an MoE they differ by ~18×.
    bytes_moved = tokens * active_params * bytes_per_param
    seconds = bytes_moved / bandwidth
    return seconds / 3600 * usd_per_hour
```

One multiplication away from Lecture 02's `bytes per token`. The cost of a
reasoning answer is this formula with `tokens` set to the trace length — and
trace length is the dial.

Two things this simple model deliberately leaves out, so you know when to stop
trusting it:

- **The KV cache.** It counts weight traffic only. At the 10,000-token traces
  above, the cache is the same order as the weights, so this *understates*
  long-trace cost — exactly the regime reasoning lives in.
- **Batching.** It prices one sequence alone. Serving many at once amortizes
  the weight read across the batch (L07), which is why the per-token cost on a
  loaded server is far below what this returns.

---

## Build it

1. Implement `reasoning_cost` and `params_tokens_tradeoff` in
   `bench/reasoning_cost.py`.
2. `uv run pytest tests/test_28b_reasoning.py -v`.
3. Compute, for Qwen3-0.6B on a 3090 at $0.25/hr: the cost of a 10,000-token
   trace, and of 64-vote majority voting over 1,000-token traces. Compare both
   to a 1-token answer.
4. Compute the `params × tokens` budget for a 32B distilled model vs a 671B
   model, and state the trade as "the 32B may think ~21× longer at equal cost".
5. **Predict first:** at what trace length does a reasoning answer's *KV cache*
   exceed the weight read? Check against Lecture 05's ~8k crossover.

---

## What you should see

**Cost per answer dominated by the trace length**, not the model size alone.

**Majority voting is linear cost for log-ish gain** — 64× tokens for ~9 points.
The diminishing return is the lesson, not the exact number.

**Distillation is the economic answer** to reasoning: a small model thinking long
undercuts a big model answering short on the `params × tokens` axis.

---

## Go deeper

- **DeepSeek-R1** ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948)): the
  source of the 77.9 → 86.7 majority-voting and the length-rises-with-accuracy
  curves.
- **[Field notes](field-notes.md)**: reasoning is test-time compute priced as
  tokens, and the KV-compression-degrades-reasoning caveat.
- **Kiely §7.2** (p.183): serving reasoning workloads, and how the tail of
  long traces stresses the scheduler differently from chat.
- **[Self-consistency](https://arxiv.org/abs/2203.11171)** (Wang et al.): the
  original "sample many, vote" method.

---

## Check yourself

1. Why is a reasoning model's "thinking" priced exactly like decode, and where
   does that price come from?
2. Majority voting buys ~9 points for 64× cost. What does that imply about the
   *first* sample vs the *last* sample?
3. What sets the ceiling on how long a single reasoning trace can run, and how
   does that differ from how much *parallel* compute you can spend?
4. Why is distilling a 671B reasoner into a 32B model an inference-cost win,
   even though the 32B does more work per answer?
5. For a *verifiable* task, why is the expensive part generating candidates and
   the cheap part scoring them — and how does that change what you optimize?

---

## Next

**[29. Contributing](29-contributing.md)**: the course ends where the field is
— reasoning, agents, and long context are exactly the open problems the open
engines are fighting over right now.

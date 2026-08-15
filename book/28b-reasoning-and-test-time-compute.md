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

against 840 MiB for a one-token answer — and its KV cache has grown to
`10,000 × 112 KiB ≈ 1.1 GiB` per step, *larger* than the weight read, so the
whole session lives in the long-context regime where the cache (L05) and its
compression (L19) dominate. "Test-time compute" is not free compute: it is
tokens, and tokens are bytes.

### Accuracy scales with tokens, sub-linearly

The reason models think at all is that more tokens buy more accuracy — but the
returns diminish. Two measured curves make the shape concrete:

- **Longer traces.** A DeepSeek-R1 write-up reports average response length
  growing from hundreds to thousands of tokens *alongside* accuracy: the reward
  signal pushes models to think longer, because thinking longer works — up to
  the point it doesn't pay.
- **Majority voting.** R1's AIME pass@1 of 77.9% becomes ~86.7% when you sample
  64 answers and vote — roughly 9 points for **64×** the decode cost. Linear
  cost, log-ish gain. That gap is the whole economics of reasoning.

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
longer. Decode cost scales with `params × tokens`, so:

```
671B model ×   1 token   =  671B-param-token
 32B model ×  ~21 tokens  =  671B-param-token
```

A 32B model can afford ~21× the thinking tokens of a 671B model at equal decode
cost (`671 / 32 ≈ 21`). Direct RL on the small model "mostly stalls" — the
reasoning ability is discovered by the big model and *copied* cheaply. This is
why the models you actually serve are distilled reasoners thinking long, not
giants answering short.

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
def reasoning_cost(params, tokens, bytes_per_param, bandwidth, usd_per_hour):
    bytes_moved = tokens * params * bytes_per_param
    seconds = bytes_moved / bandwidth
    return seconds / 3600 * usd_per_hour
```

One multiplication away from Lecture 02's `bytes per token`. The cost of a
reasoning answer is this formula with `tokens` set to the trace length — and
trace length is the dial.

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

# 12. Speculative decoding

**Build:** `engine/speculative.py::NgramSpeculator`, `verify`
**Test:** `tests/test_12_speculative.py` · **Moves:** tok/s per user, and only on the right workload
**Prereq:** [11. Chunked prefill](11-chunked-prefill.md)

---

## The problem

Every optimization so far has attacked *waste*: recomputation, padding, idle
slots, fragmentation, stalls. Decode is now efficient.

It's still **sequential**. Token N+1 needs token N: the model has to finish its
own N-th word before it can start on the next one. There is nothing inside one
user's stream to do in parallel, so each token costs one full **forward pass**
(one complete run of the model, first layer to last). From Lecture 01 each pass
re-reads all 840 MiB of weights to produce one token id, roughly 2 bytes of
output, and the weights alone are about 9 bytes of every 10 moved (215,040 of
the 232,960 MiB across a generation, 92.3%).

Batching fixed this for *aggregate* throughput, but a single user still waits one
full forward pass per token. If you want that one user to go faster, batching
doesn't help at all.

---

## The idea

The observation that makes this possible:

> Verifying N tokens costs **the same forward pass** as generating 1.

The model is memory-bound: the chip waits on bytes, and its arithmetic units sit
mostly idle. From Lecture 02, one decode step at 2k context runs at 0.79
ops:byte against a ridge of 76 on the 3090, about a percent of the machine's
available arithmetic (0.79 / 76 = 1.04%), less on bigger cards. Feed it 5
candidate tokens instead of 1 and it does 5× the arithmetic on the *same*
weight load, exactly like batching:

```
decode, 1 token:        0.79 ops:byte        <- 76 / 0.79 = 96× arithmetic
                                                   to spare
decode, 5 candidates:   5 × 0.79 = 3.95      <- 76 / 3.95 = 19× to spare
```

Still far below the ridge, so the pass ends when the bytes arrive, not when the
math finishes. The extra four tokens were nearly free.

So:

1. **Draft**, cheaply guess the next few tokens.
2. **Verify**, run the real model once over all of them.
3. **Accept** the longest correct prefix; discard the rest.

Concretely, for one generation step:

```
draft (cheap guess):      "def"   "foo"   "("   "bar"
one real forward pass:     ok      ok      ok   model wants "baz"
accepted output:          "def"   "foo"   "("   "baz"
                                3 kept + 1 corrected = 4 tokens from one pass

if all drafts had matched, the pass's last position is itself a real
prediction, so you get a 5th token, free.
```

Guess 5 and get 4 right, and you produced 4 tokens for one forward pass. Guess
badly and you produced 1: the same as before, plus the drafting cost.

**Crucially, the output is identical to normal decoding.** You accept a draft
token only when it matches what the model would have produced anyway. This isn't
an approximation, done correctly it's exact, which is why it's safe to run in
production.

### Where drafts come from

Five ways to produce the guess, differing along two axes that decide everything:
**where the draft comes from** (a separate model, the target's own weights, a
diffusion model, or the text itself) and **how much machinery it costs** to
train and serve. In roughly increasing order of setup — the last one has no
setup at all, which is why you'll build it:

**Draft model — a separate, smaller model.** The original approach (Leviathan
et al.). You keep a small model (say 0.5B) alongside the target (70B). Each
step it decodes a few tokens on its own — cheap, because decoding 0.5B
re-reads ~140× fewer weight bytes than 70B (`0.5B × 2 B = 1 GB` against
`70B × 2 B = 140 GB`), so it can grind out several drafts in the time the
target would spend on one verify. Then the target verifies them all in one
pass.

Its strengths: any off-the-shelf small model, no training. Its costs are the
two you'd expect. You serve and maintain a *second* model — its weights live in
VRAM permanently, competing with the KV cache. And its drafts are only as good
as its *agreement* with the target: two models trained on different data drift
apart, and where they disagree you get rejections. That draft-target gap is the
entire quality lever, which is why DeepSeek's **DSpec** (distillation for
speculative decoding) trains the draft model to mimic the target's own outputs —
acceptance rises sharply, because you're closing the exact gap the other
approaches only work around.

Note the drafting step itself is *still sequential*: the small model generates
token by token, and each of its tokens costs a small-model forward pass. The win
is that those passes are so much cheaper than the target's that you can afford
several per verify.

**Medusa — extra heads on the target, all at once.** Instead of a second model,
add a few extra heads to the target itself, each trained to predict the token
*i* positions ahead. The decisive difference is that all heads read the *same*
final hidden state from a single forward pass, so **drafting costs no extra
forwards**: one pass produces the drafts and, after verification, the accepted
tokens together. The cost is training the heads (a lightweight, target-specific
fine-tune). There is a ceiling: the head for position t+2 must guess from only
position t's information, so accuracy falls the further ahead you reach.
DeepSeek-V3's **MTP** (multi-token prediction) is the production form of this —
a shared lightweight block predicting the next tokens — accepting ~85–90% of
drafts, ~1.8 tokens per step.

**EAGLE — draft at the feature level.** Medusa guesses *tokens*; EAGLE guesses
the *hidden states* the tokens will produce, then runs the target's own LM head
on the guessed features to turn them into tokens. A hidden state carries far
more information than a token id — it already encodes the context and syntax the
next choice needs — so the drafts land much closer to what the target would say.
Higher acceptance than Medusa, and currently the practical default when you can
train heads. The trade: it's a trained addition to the model, and drafting still
runs a small network per position, so it isn't as free as Medusa's single-
forward drafting.

**Parallel drafting — the direction the frontier is moving.** Every method above
quietly shares one constraint: drafting is itself *autoregressive*. Even the
cheap drafters produce token `t+1` only after token `t`, so γ drafts cost γ
sequential drafting steps. They are cheap steps, but they are sequential, and
that is the same disease the whole lecture is trying to cure — one level down.

The obvious cure is a drafter that emits all γ tokens in **one parallel pass**,
conditioned on features read from the target model rather than on its own
previous guesses. If that works, the draft *cost* stops scaling with γ:
producing 16 drafts costs about what producing 4 does. The expected-tokens
formula in the next section still bounds how many drafts get *accepted*, so
acceptance stays the thing to measure — but the cost side of the trade changes
shape, and the optimal γ moves right.

This is an active research direction rather than a technique to reach for, and
the trade is the most machinery of any option here: a separate drafting model to
train, plus plumbing to feed it the target's internal states. Treat it as the
frontier, not the default — and when you meet a specific paper making this
claim, check its acceptance rate and its *drafting* cost separately, because
that is exactly where the interesting variation lives.

**N-gram / prompt lookup — no model at all.** The degenerate case, and the one
this lecture makes you build. Keep an n-gram index of the prompt and everything
generated so far; if the recent suffix has appeared before, propose whatever
followed it. Drafting is a dictionary lookup — no model, no training, no second
set of weights. It wins exactly where output *echoes* input: code completion
(the answer reuses the question's syntax), retrieval, boilerplate. And it
collapses where the text is novel: prose has no matching n-gram, acceptance near
zero, and you've added overhead for nothing. That failure is not a bug — it's
the sharpest demonstration of "the workload decides" in the lecture.

| | Draft model | Medusa / MTP | EAGLE | Parallel drafting | N-gram |
|---|---|---|---|---|---|
| Extra weights | a second model | heads on the target | trained module | a separate drafter | none |
| Training | no | yes | yes | yes | no |
| Draft mechanism | sequential small decode | one forward, parallel heads | per-position feature net | **one parallel pass** | dictionary lookup |
| Draft cost vs γ | scales with γ | scales with γ | scales with γ | **~independent of γ** | O(1) |
| Acceptance | model-dependent | good | very good | varies by method | code high, prose ~0 |
| When it wins | no training allowed | target is fine-tunable | acceptance is the goal | long γ, draft tax matters | output echoes input |

Read the last column-but-one as a research direction rather than a product you
can install today. The first three and the last are things you can actually
reach for now, and **N-gram is the one you build in this lecture** — it needs no
training, no extra weights, and it is the honest baseline every other method has
to beat.

### The metric that matters

**Acceptance rate**: what fraction of drafted tokens survive verification. Report
it alongside tok/s, always. Without it you can't tell these apart:

- Fast because acceptance is high → real win
- Fast despite low acceptance → you got lucky on batch size
- Slow despite high acceptance → verification overhead is eating the gain

Acceptance rate is not a nicety to report — it *is* the speedup, and you can
prove it.

Before the algebra, the shape. Each draft is checked in order, and the *first*
rejection ends the run — so the outcomes form a chain, not a tree of every
combination. With γ = 4:

```
  draft1   draft2   draft3   draft4
    │        │        │        │
    ✓ α      ✓ α      ✓ α      ✓ α  ──► all 4 accepted, +1 BONUS = 5 tokens
    │        │        │        │              probability α⁴
    ✗        ✗        ✗        ✗
  (1−α)    α(1−α)   α²(1−α)  α³(1−α)
    │        │        │        │
    ▼        ▼        ▼        ▼
  1 token  2 tokens 3 tokens 4 tokens
  (just    (draft1  (drafts  (drafts
   the      + the    1-2 +    1-3 +
   fix)     fix)     fix)     fix)

  note: EVERY outcome yields at least 1 token — a rejection still
  produces the corrected token, so a verify pass is never wasted
```

Read the probabilities down the ✗ row: reaching the `k`-th rejection means
accepting `k` drafts first (`α^k`) and then failing one (`1−α`), giving
`α^k(1−α)` — and that outcome is worth `k+1` tokens. The far-right branch is the
bonus: accept everything and the verify pass hands you one extra token free.

Under exact speculative decoding with γ drafts and per-token acceptance
probability α, the expected tokens per verify pass is therefore:

```
E[tokens] = 1 + α + α² + … + α^γ  =  (1 − α^(γ+1)) / (1 − α)
```

Derivation: accept *k* drafts then a rejection yields *k* + 1 tokens (the drafts
plus the correction token) with probability `α^k(1 − α)`; accept all γ and you
get γ + 1 (the bonus token) with probability `α^γ`. Summing:

```
E = Σ_{k=0}^{γ−1} (k+1)·α^k·(1−α)  +  (γ+1)·α^γ
```

That collapses to the clean geometric sum, and the collapse is worth seeing
rather than taking on faith. Do it concretely at `γ = 2`, where you can write
every term out:

```
E = 1·(1−α)          k=0: reject immediately, 1 token
  + 2·α·(1−α)        k=1: one draft accepted, then a rejection
  + 3·α²             all γ=2 accepted, plus the bonus token

  = (1 − α) + (2α − 2α²) + 3α²        expand each product

  = 1 − α + 2α − 2α² + 3α²            drop the brackets

  = 1 + α + α²                        collect: (−α+2α)=α, (−2α²+3α²)=α²
```

Every term cancels except the geometric one. The same cancellation happens at
any γ — each `(k+1)α^k` contributes `+(k+1)α^k`, and the `−(k+1)α^(k+1)` it
drags along is exactly cancelled by the next term's `+(k+2)α^(k+1)` minus one
copy, leaving a single `α^(k+1)`. It telescopes:

```
E = Σ_{k=0}^{γ−1} (k+1)·α^k·(1−α)  +  (γ+1)·α^γ  =  1 + α + α² + … + α^γ
```

And a geometric series has a closed form, which is where the `(1 − α^(γ+1))/(1 − α)`
above comes from:

```
1 + α + … + α^γ  =  (1 − α^(γ+1)) / (1 − α)        for α < 1
```

At `α = 0.8, γ = 4`: `1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.36` tokens per verify
— a 3.36× wall-clock win over one token per pass. At `α = 0.2`, the same drafts
buy `1.25×`: barely above doing nothing. That gap is the entire lecture, and
it's why this is one of the few techniques that moves *per-user* latency, which
batching (Lecture 07) cannot touch.

Now notice that the formula you just derived has an awkward property. `E[tokens]
= 1 + α + … + α^γ` is **strictly increasing in γ** — every extra draft adds
another positive `α^k` term. Taken literally, it says: draft 50 tokens, draft
500, the expected yield only goes up.

Reality disagrees. From the [field notes](field-notes.md), an operator running
2×3090 found the documented 3 draft tokens beaten by 5, and above 5 performance
got measurably **worse**. So the model and the measurement contradict each other,
and the model is the one that is wrong — or rather, incomplete.

**What the formula leaves out is the cost of drafting.** It counts tokens
*produced per verify pass* and says nothing about what the pass costs. Put both
sides in:

```
                    E[tokens]           1 + α + … + α^γ
  speedup  ≈  ───────────────────  =  ───────────────────
              1 verify + γ drafts       1 + γ·c

  where c = cost of one draft step ÷ cost of one target forward pass
```

The numerator grows, but it grows by ever-smaller amounts — the terms are
`α^k`, shrinking geometrically. The denominator grows **linearly**, forever. A
ratio whose numerator saturates and whose denominator doesn't must eventually
turn over, and that turning point is the peak:

```
  E[tokens]     ▁▃▅▆▇▇▇▇▇  saturates at 1/(1−α)
  draft cost    ▁▂▃▄▅▆▇█▉  grows without limit
                ─────────
  speedup       ▁▃▅▆▇▆▅▃▁  rises, peaks, falls
                      ▲
                   the peak the field notes measured at γ=5
```

Where the peak sits depends on both α and `c`: a cheaper drafter (small `c`)
pushes it right, a lower acceptance rate pulls it left. That is also exactly why
the parallel-drafting direction above matters — it attacks `γ·c`, flattening the
denominator, which moves the peak right.

Past the peak you are paying to verify tokens that get thrown away. **Measure
your own peak; do not inherit someone else's γ.**

### The workload decides everything

This is the lesson, and it's why `bench/workloads.py` ships two workloads for it:

| Workload | Why | Acceptance |
|---|---|---|
| **`code_completion`** | output echoes input; syntax is predictable | high |
| **`prose`** | novel text; n-grams have nothing to match | low |

Real numbers from the field notes: **+29% average, +50% on code.** Benchmark only
on code and you'll oversell it; only on prose and you'll conclude it's useless.

There's also a throughput interaction: speculation consumes spare compute, so it
helps most at **low batch sizes** where capacity is idle. Under heavy load your
batch is already using that capacity, and speculation can *hurt*. Latency
optimization and throughput optimization pull against each other here.

---

## The code

```python
class NgramSpeculator:
    """Draft by looking up recent n-grams in the text so far."""

    def __init__(self, n: int = 3, max_draft_tokens: int = 8):
        self.n = n
        self.max_draft_tokens = max_draft_tokens

    def propose(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) < self.n:
            return []
        suffix = tuple(token_ids[-self.n:])

        # Search backwards -- the most recent match is the best predictor.
        for i in range(len(token_ids) - self.n - 1, -1, -1):
            if tuple(token_ids[i:i + self.n]) == suffix:
                start = i + self.n
                return token_ids[start:start + self.max_draft_tokens]
        return []
```

Verification, and the part people get wrong:

```python
def verify(target_logits, draft_tokens):
    """Accept the longest correct prefix, then ALWAYS take one bonus token.

    target_logits[i] is the model's prediction given draft_tokens[:i].
    """
    accepted = []
    for i, drafted in enumerate(draft_tokens):
        predicted = int(target_logits[i].argmax())
        if predicted != drafted:
            accepted.append(predicted)   # the correction is a real token
            return accepted, len(accepted)
        accepted.append(drafted)

    # All drafts accepted -- the final logits give a free extra token.
    accepted.append(int(target_logits[len(draft_tokens)].argmax()))
    return accepted, len(accepted)
```

Two subtleties worth stating plainly:

**A rejection still yields a token.** When draft `i` is wrong, the model's own
prediction at that position is correct by definition, keep it. Miss this and
speculation can be *slower* than not speculating, because a full rejection would
produce nothing.

**Full acceptance yields a bonus token.** With N drafts verified you get N+1
tokens: the logits after the last accepted draft are a genuine prediction. Free.

**Stop at the first rejection.** Everything after a wrong token was conditioned on
it, so it's invalid regardless of what it says.

> For sampling (`T > 0`), exact equivalence requires **rejection sampling** with
> the acceptance probability from the Leviathan et al. paper, not simple argmax
> comparison. Greedy is a fair place to start, but if you claim "identical
> output distribution" while sampling, you need the real algorithm.

---

## Build it

1. Implement `NgramSpeculator.propose` and `verify` in `engine/speculative.py`.
2. Wire draft → verify → accept into your decode loop.
3. `uv run pytest tests/test_12_speculative.py -v`, **speculative greedy output
   must exactly match non-speculative greedy.** Not "similar." Identical.
4. Measure the contrast:

```bash
uv run python book/code/spec_bench.py
```

Runs `code_completion` and `prose`. **Report acceptance rate for both.**

5. Sweep `max_draft_tokens` over 2/4/8/16 and find where more speculation stops
   paying. Compare your answer to the field notes' 5.

---

## What you should see

**Solid gains on `code_completion`**: repetitive syntax, high acceptance.

**Little or nothing on `prose`**: novel text, n-grams miss. That's not a failure;
it's the lesson.

**A peak in the draft-length sweep**, then decline. Past the peak you're paying to
verify tokens that get thrown away.

**Bigger wins at batch size 1** than under load.

---

## Go deeper

- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)**
  (Leviathan et al., 2022): the original. §2.3's rejection-sampling proof is what
  makes this exact rather than approximate; read it before claiming distributional
  equivalence.
- **[EAGLE](https://arxiv.org/abs/2401.15077)**: feature-level drafting, the
  current practical default.
- **[Medusa](https://arxiv.org/abs/2401.10774)**: multiple decoding heads.
- **Distillation for speculative decoding**: train the draft model to mimic the
  target's distribution and acceptance goes up, because acceptance *is* the
  agreement between the two models. A productive line of work and the most
  reliable way to improve α for a given draft-model size.
- **Parallel / non-autoregressive drafting**: the frontier direction described
  above, where the drafter emits γ tokens in one pass instead of γ. When
  evaluating any paper here, separate the two claims — acceptance rate, and
  drafting cost per token — because the headline speedup is a product of both
  and papers vary in which one they actually improve.
- **Kiely §5.2–5.2.4** (p.129–136), all four approaches compared, including the
  n-gram/lookahead variant you built.
- **[Field notes](field-notes.md)**: docs said 3 draft tokens, measurement said 5,
  and >5 was worse. Tune on *your* workload.

---

## Check yourself

1. Why does verifying 5 tokens cost about the same as generating 1?
2. Why is a *rejected* draft position still a valid token?
3. Acceptance is 90% but tok/s barely moved. What's happening?
4. Speculation helps at batch 1 and can hurt at batch 64. Why?
5. Your acceptance is 65% on code and 15% on prose. What single number would you
   quote to someone asking "how much does speculative decoding help?"

??? question "Done? Tap to check — three clicks to the full answer"
    ??? question "Still stuck? Show a hint for each"
        1. Decode leaves arithmetic idle; verification fills that idle headroom.
        2. The model's own prediction is a real token by definition.
        3. Overhead, not acceptance — and there are two other cases worth naming.
        4. Idle capacity, not context decay: speculation is a latency play.
        5. There is no single number, which is the point.

        ??? question "Show the answers"
            1. Verifying 5 candidates does 5× the arithmetic on the same
               840 MiB weight load. Decode runs at 0.79 ops:byte against a
               ridge of 76 — ~96× arithmetic headroom — so the extra math hides
               in time the chip was already waiting on bytes. The pass still
               ends when the bytes arrive.
            2. When draft *i* is wrong, the model's prediction at that position
               is, by definition, what the model would have generated. Keep it:
               drop it and a full rejection produces nothing, which makes
               speculation *slower* than not speculating.
            3. High acceptance with flat tok/s means the verify/draft overhead
               is eating the gain. Use the three-way test: fast + high
               acceptance = real win; fast + low acceptance = lucky batch size;
               slow + high acceptance = overhead.
            4. Not context decay. Speculation consumes *spare arithmetic*. At
               batch 1 the decode's compute units sit idle, so verification
               rides along nearly free. At batch 64 the machine is already
               busy, there's no idle capacity left, and the extra verification
               work costs more than it saves — the latency-vs-throughput
               tradeoff pulling against itself.
            5. No honest single number. Acceptance is 65% on code and 15% on
               prose, so the win is workload-dependent — quote both numbers
               with the workload, or don't quote one at all.

---

## Next

**[12b. Structured output and adapters](12b-structured-output.md)**: what comes
out, not just how fast: guided decoding, tool calling, LoRA.

Lighter on implementation than the last few. The one thing to build is a
**logit-processor hook** in `engine/sampling.py`: the extension point all three
features hang off.

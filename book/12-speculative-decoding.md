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

Four ways to produce the guess, differing along two axes that decide everything:
**where the draft comes from** (a separate model, the target's own weights, or
the text itself) and **how much machinery it costs** to train and serve. In
roughly increasing order of setup:

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

**N-gram / prompt lookup — no model at all.** The degenerate case, and the one
this lecture makes you build. Keep an n-gram index of the prompt and everything
generated so far; if the recent suffix has appeared before, propose whatever
followed it. Drafting is a dictionary lookup — no model, no training, no second
set of weights. It wins exactly where output *echoes* input: code completion
(the answer reuses the question's syntax), retrieval, boilerplate. And it
collapses where the text is novel: prose has no matching n-gram, acceptance near
zero, and you've added overhead for nothing. That failure is not a bug — it's
the sharpest demonstration of "the workload decides" in the lecture.

| | Draft model | Medusa / MTP | EAGLE | N-gram |
|---|---|---|---|---|
| Extra weights | a second model | heads on the target | trained module | none |
| Training | no | yes | yes | no |
| Draft mechanism | sequential small decode | one forward, parallel heads | per-position feature net | dictionary lookup |
| Acceptance | model-dependent | good | **best** | code high, prose ~0 |
| When it wins | no training allowed | target is fine-tunable | acceptance is the goal | output echoes input |

### The metric that matters

**Acceptance rate**: what fraction of drafted tokens survive verification. Report
it alongside tok/s, always. Without it you can't tell these apart:

- Fast because acceptance is high → real win
- Fast despite low acceptance → you got lucky on batch size
- Slow despite high acceptance → verification overhead is eating the gain

Acceptance rate is not a nicety to report — it *is* the speedup, and you can
prove it. Under exact speculative decoding with γ drafts and per-token
acceptance probability α, the expected tokens per verify pass is:

```
E[tokens] = 1 + α + α² + … + α^γ  =  (1 − α^(γ+1)) / (1 − α)
```

Derivation: accept *k* drafts then a rejection yields *k* + 1 tokens (the drafts
plus the correction token) with probability `α^k(1 − α)`; accept all γ and you
get γ + 1 (the bonus token) with probability `α^γ`. Summing:

```
E = Σ_{k=0}^{γ−1} (k+1)·α^k·(1−α)  +  (γ+1)·α^γ  =  1 + α + α² + … + α^γ
```

At `α = 0.8, γ = 4`: `1 + 0.8 + 0.64 + 0.512 + 0.4096 = 3.36` tokens per verify
— a 3.36× wall-clock win over one token per pass. At `α = 0.2`, the same drafts
buy `1.25×`: barely above doing nothing. That gap is the entire lecture, and
it's why this is one of the few techniques that moves *per-user* latency, which
batching (Lecture 07) cannot touch.

γ is not free either — a rejected draft is wasted work — so the peak sits at
moderate γ. From the [field notes](field-notes.md), an operator running 2×3090
found the documented 3 draft tokens beaten by 5, and above 5, performance got
measurably **worse**: past the peak you're paying to verify tokens that get
thrown away.

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
- **[DSpec](https://arxiv.org/abs/2406.14846)** (Zhou et al., 2024): distillation
  for speculative decoding — train the draft model to mimic the target and
  acceptance goes up. DeepSeek's contribution to the draft-model line.
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

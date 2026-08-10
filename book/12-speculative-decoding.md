# 12 — Speculative decoding

**Build:** `engine/speculative.py::NgramSpeculator`, `verify`
**Test:** `tests/test_12_speculative.py` · **Moves:** tok/s per user — and only on the right workload
**Prereq:** [11 — Chunked prefill](11-chunked-prefill.md)

---

## The problem

Every optimization so far has attacked *waste*: recomputation, padding, idle
slots, fragmentation, stalls. Decode is now efficient.

It's still **sequential**. Token N+1 needs token N. One token per forward pass,
and each pass drags all 840 MiB of weights through memory to produce ~2 bytes of
output.

Batching fixed this for *aggregate* throughput — but a single user still waits one
full forward pass per token. If you want that one user to go faster, batching
doesn't help at all.

---

## The idea

The observation that makes this possible:

> Verifying N tokens costs **the same forward pass** as generating 1.

The model is memory-bound. Feed it 5 candidate tokens instead of 1 and it does 5×
the arithmetic on the *same* weight load — nearly free, exactly like batching.

So:

1. **Draft** — cheaply guess the next few tokens.
2. **Verify** — run the real model once over all of them.
3. **Accept** the longest correct prefix; discard the rest.

Guess 5 and get 4 right, and you produced 4 tokens for one forward pass. Guess
badly and you produced 1 — the same as before, plus the drafting cost.

**Crucially, the output is identical to normal decoding.** You accept a draft
token only when it matches what the model would have produced anyway. This isn't
an approximation — done correctly it's exact, which is why it's safe to run in
production.

### Where drafts come from

**Draft model** — a small model (0.5B guessing for 70B). Good acceptance, but you
run and maintain a second model.

**Medusa** — extra heads on the target model predicting several positions ahead.
No separate model, but requires training.

**EAGLE** — predicts at the *feature* level rather than the token level, using
hidden states. Higher acceptance than Medusa; the current default when you can
train heads.

**N-gram / prompt lookup** — no model at all. Build an n-gram map from the prompt
and the text so far; if the recent suffix appeared before, propose whatever
followed it. **This is what you'll implement**, because it needs no training and
the entire draft/verify/accept loop is visible in a few dozen lines.

### The metric that matters

**Acceptance rate** — what fraction of drafted tokens survive verification. Report
it alongside tok/s, always. Without it you can't tell these apart:

- Fast because acceptance is high → real win
- Fast despite low acceptance → you got lucky on batch size
- Slow despite high acceptance → verification overhead is eating the gain

More speculation is *not* more speed. Rejected drafts cost real compute, and past
a point you lose. From the [field notes](field-notes.md), an operator running
2×3090 found the documented 3 draft tokens beaten by 5 — measured via mean
acceptance length — and above 5, performance got measurably **worse**.

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
prediction at that position is correct by definition — keep it. Miss this and
speculation can be *slower* than not speculating, because a full rejection would
produce nothing.

**Full acceptance yields a bonus token.** With N drafts verified you get N+1
tokens: the logits after the last accepted draft are a genuine prediction. Free.

**Stop at the first rejection.** Everything after a wrong token was conditioned on
it, so it's invalid regardless of what it says.

> For sampling (`T > 0`), exact equivalence requires **rejection sampling** with
> the acceptance probability from the Leviathan et al. paper, not simple argmax
> comparison. Greedy is a fair place to start — but if you claim "identical
> output distribution" while sampling, you need the real algorithm.

---

## Build it

1. Implement `NgramSpeculator.propose` and `verify` in `engine/speculative.py`.
2. Wire draft → verify → accept into your decode loop.
3. `uv run pytest tests/test_12_speculative.py -v` — **speculative greedy output
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

**Solid gains on `code_completion`** — repetitive syntax, high acceptance.

**Little or nothing on `prose`** — novel text, n-grams miss. That's not a failure;
it's the lesson.

**A peak in the draft-length sweep**, then decline. Past the peak you're paying to
verify tokens that get thrown away.

**Bigger wins at batch size 1** than under load.

---

## Go deeper

- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)**
  (Leviathan et al., 2022) — the original. §2.3's rejection-sampling proof is what
  makes this exact rather than approximate; read it before claiming distributional
  equivalence.
- **[EAGLE](https://arxiv.org/abs/2401.15077)** — feature-level drafting, the
  current practical default.
- **[Medusa](https://arxiv.org/abs/2401.10774)** — multiple decoding heads.
- **Kiely §5.2–5.2.4** (p.129–136) — all four approaches compared, including the
  n-gram/lookahead variant you built.
- **[Field notes](field-notes.md)** — docs said 3 draft tokens, measurement said 5,
  and >5 was worse. Tune on *your* workload.

---

## Check yourself

1. Why does verifying 5 tokens cost about the same as generating 1?
2. Why is a *rejected* draft position still a valid token?
3. Acceptance is 90% but tok/s barely moved. What's happening?
4. Speculation helps at batch 1 and can hurt at batch 64. Why?
5. Your acceptance is 65% on code and 15% on prose. What single number would you
   quote to someone asking "how much does speculative decoding help?"

That last one has no honest single answer, which is the point.

---

## Next

**[12b — Structured output and adapters](12b-structured-output.md)** — what comes
out, not just how fast: guided decoding, tool calling, LoRA.

Lighter on implementation than the last few. The one thing to build is a
**logit-processor hook** in `engine/sampling.py` — the extension point all three
features hang off.

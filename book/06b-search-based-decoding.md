# 06b. Search-based decoding

**Build:** `engine/beam_search.py` · **Test:** `tests/test_06b_beam.py`
**Moves:** nothing you ship — it moves *understanding* of why sampling beat search
**Prereq:** [06. Sampling](06-sampling.md)

---

## The problem

Lecture 06 decoded by argmax ("greedy") or by rolling a weighted die
("sampling"). There is a third family, and it was the default for a decade:
**search** — keep many partial sequences alive and pick the best finished one.

The book has avoided it until now, and this lecture is about *why*, because the
reason is a real insight, not a taste. Greedy decoding picks the most likely
token **per step**. Search targets the most likely **sequence**. Those are
different things, and the difference is where all the subtlety lives.

---

## The idea

### The per-token mode is not the sequence mode

One token is easy: the most likely next token is `argmax(logits)`, exactly.
The most likely *sequence* is not. Over a vocabulary of `V` and a length of `T`
there are `V^T` completions; the per-step argmax does **not** pick the argmax
of that space, because the two goals disagree the moment one slightly-less-
likely token leads to a far-more-likely continuation:

```
step 0:  token 0 (logp 0.6)   token 1 (logp 0.4)
step 1:  after 0 -> [0.1, 0.9]      after 1 -> [9.0, 8.0]
```

Greedy walks into the trap: it takes token 0 at step 0 (0.6 > 0.4), then token 1
(0.9), for a path score of `0.6 + 0.9 = 1.5`. The best path is `1 then 0` at
`0.4 + 9.0 = 9.4`. A low-probability first step can be the entrance to a
high-probability future, and argmax can't see past the current step. This is the
"splitting point", and it is why search exists.

### Beam search

Beam search keeps the `K` highest-scoring partial sequences alive at each step
instead of one. Each step, every surviving path is extended by every token, the
`K` best extensions are kept, and the rest are dropped:

```
K=2:  step 0:  keep [0] (0.6), [1] (0.4)
      step 1:  [0,0]=0.7  [0,1]=1.5  [1,0]=9.4  [1,1]=8.4   -> keep [1,0], [1,1]
```

The path greedy threw away (token 1 first) survived long enough to reveal its
9.4. That is the entire mechanism. `K` is the **beam width**; `K=1` *is* greedy,
and `K=V^T` is exhaustive search, with beam search the entire spectrum between.

### The length problem

Path scores are sums of log-probabilities, and every log-probability is
negative. So **more tokens means a more negative score**, always: a completed
generation's score is monotonically non-increasing in its length. Left
unchecked, the empty string is the argmax.

The fix is **length normalization** — divide the score by `length^alpha`:

```
score = sum(log p_t) / |y|^alpha
```

With `alpha=1` a five-token path averaging -1.0 per token scores `-5/5 = -1.0`,
beating a one-token path at `-2.0`. Without it, the one-token path wins despite
being the *worse average* decision. (Real tokenizers complicate the exact
form — HuggingFace uses `((5 + len) / 6)^alpha` — but the shape is this.)

### A* and best-first search

Beam search is one instance of a bigger idea. Think of decoding as searching a
tree of prefixes, and score each node with `f = g + h`: `g` is the log-prob of
the prefix so far, `h` is an estimate of the rest. **A\*** pops the best-scoring
frontier node first, and with an admissible `h` (never overestimates) it is
guaranteed to find the argmax. **Best-first** is A* with `h = 0` (Dijkstra over
tokens). Beam search is the same tree search with a fixed frontier size `K`
instead of a priority order — cruder, but with a *bounded* memory footprint,
which is the property that actually matters for a memory-bound operation.

### Why the book (and production) mostly skips search

Three reasons, all grounded in things you already know:

1. **The likelihood trap.** Search maximizes the same quantity greedy maximizes,
   only more thoroughly. From Lecture 06, that quantity — likelihood — is
   *anti-correlated* with what humans prefer: the highest-probability
   continuations are short, repetitive, and generic. Beam search with a large
   `K` converges on them with confidence. This is documented ("the curse of beam
   search": bigger beam often *worse* downstream accuracy).

2. **The cost model.** `K` beams run as one **batch-K** forward pass. Decode is
   memory-bound, so per-step weight traffic barely moves — one step re-reads 840
   MiB regardless of `K` — but the KV cache grows to `112 KiB × context × K`.
   Beam search buys its candidates with *memory*, and at `K=8`, 2k context, that
   is 8× the KV cache for the same wall-clock per step.

3. **A verifier beats a beam.** When you *can* score a finished answer (the
   best-of-N / reward-model pattern, Lecture 28b), sampling many candidates and
   scoring them is strictly more flexible than maintaining one beam, because the
   scorer sees whole answers, not prefixes.

Search still earns its place where a strong, cheap per-step score exists and the
mode genuinely is what you want — machine translation, structured transcription,
anything with a near-deterministic correct answer.

---

## The code

```python
def beam_search(step_scorer, max_len, beam_width):
    beams = [([], 0.0)]                        # (path, score)
    for _ in range(max_len):
        candidates = []
        for path, score in beams:
            for token, logp in enumerate(step_scorer(path)):
                candidates.append((path + [token], score + logp))
        candidates.sort(key=lambda x: -x[1])   # highest score first
        beams = candidates[:beam_width]        # keep only the top K
    return beams[0][0]
```

Nothing else. The `step_scorer` returns log-probabilities for the next token
given the path so far; in a real engine it's `model(path).logits[-1]`.

---

## Build it

1. Implement `beam_search` and `normalize_length` in `engine/beam_search.py`.
2. `uv run pytest tests/test_06b_beam.py -v`. The splitting-point test is the
   point: your beam must find `[1, 0]` where greedy finds `[0, 1]`.
3. Extend the scorer idea to length: show that without `normalize_length`, a
   1-token completion always beats a 5-token one, and that normalization flips
   it.
4. (Optional, GPU) Swap `beam_search` into your engine for one experiment: run
   `beam_width=4` against greedy on a small prompt, and measure (a) KV cache
   growth and (b) whether the output is better, worse, or just *different*.

---

## What you should see

**Beam finds the path greedy walks past**, on the splitting-point grid.

**Length normalization flips the short-vs-long preference.** This is the one
bug that shows up in real code: an unnormalized scorer silently prefers the
empty completion.

**Larger beam width is not monotonically better**, once you hook it to a real
task — the likelihood trap is real, and it is the whole reason the book decodes
by sampling with a verifier instead.

---

## Go deeper

- **[The Curse of Beam Search](https://arxiv.org/abs/1904.09751)**
  (also Holtzman et al., 2019): larger beams *reduce* downstream quality on
  likelihood-trained models. The empirical case against search.
- **[A* sampling](https://arxiv.org/abs/2103.13923)** (Luong et al.): search with
  a probabilistic twist. Useful middle ground.
- **"If beam search is the answer, what was the question?"** — the framing that
  search's value depends entirely on whether the mode is what you want.
- **HuggingFace `generate` docs** on `num_beams`, `length_penalty`, and the
  `((5+len)/6)^alpha` normalization — the production details.

---

## Check yourself

1. Greedy finds `[0, 1]` (score 1.5) but the argmax path is `[1, 0]` (9.4).
   Why doesn't greedy see it?
2. Why does the raw path score always prefer a shorter completion, and what does
   `length_penalty` do about it?
3. Beam width `K=8` at 2k context: how much KV cache does a single sequence need,
   and why is that the real cost rather than the compute?
4. Why does increasing beam width sometimes make *worse* text?
5. When would you actually choose beam search over sampling for an LLM?

---

## Next

**[07. Static batching](07-static-batching.md)**: back to the engine, and the
first place the KV cache becomes a shared, contended resource.

Search is a detour from the main line — if you're in a hurry, skip it and return
when you meet best-of-N in Lecture 28b.

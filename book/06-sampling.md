# 06. Sampling

**Build:** `engine/sampling.py::sample` · **Test:** `tests/test_06_sampling.py`
**Moves:** nothing: this is correctness infrastructure · **Prereq:** [05](05-kv-cache.md)

---

## The problem

Every generation so far has been greedy: take the highest-logit token, always.
Real serving isn't, users send `temperature`, `top_p`, `top_k`, and expect them
to mean something.

Sampling is cheap to implement and easy to get subtly wrong. The bugs don't
crash; they produce text that looks fine and is quietly incorrect. Worse, they
make everything downstream harder to debug, because you can no longer tell "my
scheduler is broken" from "my sampler is random."

Short lecture. It's here because the rest of the book depends on it.

---

## The idea

The model gives you **logits**, one unnormalized score per vocabulary token.
Sampling turns that vector into one chosen token.

Think of a weighted die. The faces are the **vocabulary**, every token the
model knows how to output, and each face's chance of coming up is set by its
logit: likely tokens land more often. Greedy decoding, what you've done so
far, is "always take the most likely face". Sampling is "roll the die". Four
knobs reshape the die, applied in a specific order.

### Temperature

Divide the logits before softmax:

```python
logits = logits / temperature
```

- `T < 1` sharpens the distribution (the list of chances over the vocabulary,
  always adding up to 1), more confident, more repetitive.
- `T > 1` flattens it, more diverse, more likely to be incoherent.
- `T = 0` is a special case: it means **argmax** (pick the highest score,
  always), not division by zero. Guard it.

??? question "Why not just use a very small temperature instead of special-casing 0?"
    Two reasons. First, it's a literal division by zero, logits/T gives NaN.
    Second, any finite T is still a probability distribution: even at `T=1e-9`
    the sampler *could* pick token #2, the probabilities are just tiny.
    "Tiny T ≈ greedy" is never exactly deterministic, and this lecture's
    whole reason for having greedy is the deterministic test oracle. So
    argmax gets its own code path that is guaranteed deterministic, instead
    of a distribution so peaked that float jitter decides.

    [Full answer in the Q&A](qa.md#why-is-temperature-0-special-cased-instead-of-a-very-small-divisor)

??? question "Why is T=1 the only temperature that samples the model's true distribution?"
    Temperature reshapes the logits *before* softmax, so `T≠1` draws from a
    tempered distribution, `q(x) ∝ p(x)^(1/T)`, not from `p` itself. Sample at
    `T=0.7` and your averages converge to a sharpened model, not the model.
    `T=1` is the one setting where the die's faces are the model's actual
    probabilities — the only unbiased way to *measure* what the model believes,
    as opposed to steering it. (In practice `T>1` is almost never used: it
    pushes the model past "diverse" into "off the rails".)
    [Full answer](qa.md#why-is-t1-the-only-temperature-that-samples-the-models-true-distribution)

### Top-k

Keep the `k` highest-probability tokens, zero the rest, renormalize. Blunt but
effective: the tail of a 150k-token vocabulary is mostly garbage that
collectively holds non-trivial probability mass.

### Top-p (nucleus)

Sort by probability descending, take tokens until the cumulative sum exceeds `p`,
discard the rest. Adaptive where top-k is fixed: when the model is confident, the
nucleus is 2 tokens; when uncertain, it might be 200.

**Always keep at least one token.** If the top token already exceeds `p`, a naive
cumulative-sum filter can mask everything and leave you sampling from nothing.
This is the single most common top-p bug.

### Repetition penalty

Divide (or multiply) the logits of already-generated tokens to discourage loops.
The goal: a token that's already been generated should be *less* likely to be
picked again. Since a higher logit means more likely, the penalty has to push
every penalized logit **away from zero**:

```
positive logit, divide by penalty:   5.0 / 1.2  =  4.17   (less favorable)  ✓
negative logit, multiply by penalty: -3.0 × 1.2  =  -3.6  (even less favorable) ✓
zero: 0 / 1.2 = 0 × 1.2 = 0          (neutral either way)
```

Note the asymmetry: for a positive logit you divide, for a negative one you
multiply. A naive "divide everything" turns the penalty into a *reward* for
negative-logit tokens: `-3.0 / 1.2 = -2.5`, which is closer to zero, hence
more likely, the exact opposite of punishment. Another classic bug.

??? question "Why multiply the negatives instead of dividing them?"
    Softmax is monotone in the logit: closer to zero = more probable. Dividing
    a negative number moves it toward zero, which *raises* its probability.
    Multiplying a negative by something > 1 moves it further away, which is
    what "penalize" actually means. [Worked example in the
    Q&A](qa.md#why-does-the-repetition-penalty-divide-positive-logits-but-multiply-negative-ones)

### Order matters

```
repetition penalty → temperature → top-k → top-p → sample
```

Penalties act on raw logits. Temperature reshapes the distribution. Then you
truncate. Applying top-p *before* temperature gives different results, and it's
not what anyone means by these parameters.

---

## The code

```python
def sample(logits, params, prev_tokens=None):
    # 1. repetition penalty, on raw logits
    if params.repetition_penalty != 1.0 and prev_tokens is not None:
        for t in set(prev_tokens):
            if logits[t] > 0:
                logits[t] /= params.repetition_penalty
            else:
                logits[t] *= params.repetition_penalty   # sign matters!

    # 2. greedy is a special case, not T=0 division
    if params.temperature == 0.0:
        return int(logits.argmax())

    # 3. temperature
    logits = logits / params.temperature

    # 4. top-k
    if params.top_k > 0:
        kth = torch.topk(logits, min(params.top_k, logits.size(-1))).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    # 5. top-p
    if params.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative - probs > params.top_p   # keep the one that crosses
        remove[0] = False                            # ALWAYS keep the top token
        logits = logits.masked_fill(
            remove.scatter(0, sorted_idx, remove), float("-inf")
        )

    # 6. sample
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1))
```

Two lines deserve attention.

`cumulative - probs > top_p` rather than `cumulative > top_p`: this keeps the
token that *crosses* the threshold. Otherwise with `top_p=0.9` and a top token at
0.95, you'd remove everything.

`remove[0] = False` is the guard. Belt and braces, with the shifted comparison
it's redundant, but this is exactly the code that gets refactored later by someone
who doesn't know why the shift is there.

---

## Determinism is a feature

The reason this lecture sits before the scheduler:

> **Greedy decoding is deterministic. That makes it a test oracle.**

Every optimization from here, batching, paging, prefix caching, speculative
decoding, must not change the output. With `temperature=0` you can assert
*exact* token equality and know that any difference is a bug.

Lose determinism and you lose the ability to distinguish "my paged attention has
an indexing bug" from "sampling was different this time." Debugging a scheduler
without a deterministic oracle is genuinely miserable.

So: **seed everything, test in greedy mode, and treat any nondeterminism in the
greedy path as a bug**, not as noise to average away.

??? question "Why does greedy decoding produce repetitive, degenerate text?"
    Because picking the most-likely token *each step* does not pick the
    most-likely *sequence*. Picture a biased coin, 60% heads. The single most
    likely ordered outcome is "100 heads", at `0.6¹⁰⁰ ≈ 6.5×10⁻²³`. But "about
    60 heads" is the *typical* outcome: any one such sequence is only
    `0.6⁶⁰·0.4⁴⁰ ≈ 5×10⁻³⁰` (a billion times less likely than "100 heads"),
    yet there are `C(100,60) ≈ 1.4×10²⁸` of them, so nearly all the probability
    mass lives there. Argmax chases the point of highest density — a string so
    unlikely no sampling run would ever produce it — which is why likelihood-
    maximizing decoders loop and repeat (Holtzman et al. call it the "repetition
    trap"). Sampling wanders through the bulk of the distribution instead of
    pinning the spike.
    [Full answer](qa.md#why-does-greedy-decoding-produce-repetitive-degenerate-text)

??? question "Why greedy/sampling instead of beam search?"
    Beam search keeps `K` partial sequences alive each step, targeting the
    *mode of the whole sequence*, not the per-token mode. Two things make it the
    wrong default for an LLM. First, that sequence mode is exactly the
    degenerate likelihood trap above — beam search maximizes the same quantity
    greedy does, only more thoroughly. Second, it costs you on an axis you now
    understand: `K` beams run as one **batch-K** forward pass, so per-step weight
    traffic is ~unchanged (decode is memory-bound, one step re-reads 840 MiB
    regardless of `K`), but the KV cache is `112 KiB × context × K`. Frontier
    serving has largely dropped it — a verifier (best-of-N) or plain sampling
    beats it for most tasks. Beam/A* still earn their keep where a strong
    scorer exists (machine translation, exact decoding).
    [Full answer](qa.md#why-greedysampling-instead-of-beam-search)

---

## Build it

1. Implement `sample()` and `SamplingParams` in `engine/sampling.py`.
2. `uv run pytest tests/test_06_sampling.py -v`
3. Wire it into `generate_cached`, greedy when `temperature=0`, sampled
   otherwise. Every existing test must still pass, because they all use greedy.
4. Sanity check by hand: generate the same prompt at `T=0`, `T=0.7`, `T=2.0`.
   Read the outputs. `T=2.0` should be visibly unhinged.

---

## Go deeper

- **[The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)**
  (Holtzman et al., 2019), introduced nucleus sampling. The figures showing why
  pure likelihood maximization produces repetitive text are worth the read.
- **Beyond top-k/top-p.** There's a whole family of truncation samplers the
  book skips: **epsilon sampling** (drop tokens below a probability floor),
  **η-sampling** (threshold from the local entropy), **min-p** (relative to the
  top token), and **locally typical sampling** (keep tokens whose log-prob is
  near the entropy; it *raises* perplexity on purpose). The Q&A unpacks the
  family.
- **vLLM `vllm/sampling_params.py`**: the production surface area. Note how many
  parameters exist beyond these four, and that they're applied in a defined order
  for the same reason.
- **Kiely §1.3.1** (p.31), evaluation, and why sampling settings make
  benchmarking harder than it looks.

---

## Check yourself

1. Why is `temperature=0` a special case rather than just a very small divisor?
2. `top_p=0.9`, and the model's top token has probability 0.95. What should
   happen, and what does a naive `cumulative > 0.9` filter do instead?
3. You're debugging your scheduler and outputs differ between runs. Why is that
   much harder to diagnose if you were sampling with `T=0.8`?

---

## Next

**[07. Static batching](07-static-batching.md)**: serve more than one request,
and meet a new kind of waste.

```bash
uv run python book/code/batching_waste.py    # run before reading
```

Same code, 0% waste on one workload and 61% on another. **Record the padding
waste, not just throughput**, that number is what L08 eliminates.

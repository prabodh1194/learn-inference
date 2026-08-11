# 03. Naive generation

**Build:** `engine/model.py::load`, `engine/generate.py::generate_naive`
**Test:** `tests/test_03_generation.py` · **Demo:** `book/code/recomputation.py`
**Moves:** your first real number, tok/s and the shape of the curve
**Prereq:** [02. Arithmetic intensity](02-arithmetic-intensity.md)

---

## The problem

Time to write code. You'll write the *slowest reasonable* generation loop, no
cache, no batching, and measure exactly how badly it scales.

This is deliberate. Everything in Part II is a fix for something you're about to
feel. Skip this and the KV cache is a fact you memorized; do it and the KV cache
is the obvious solution to a problem that annoyed you.

---

## See it

```bash
uv run python book/code/recomputation.py
```

Generating 512 tokens from a 64-token prompt, with no cache:

```
  K/V vectors computed : 163,584
  actually needed      :     512
  thrown away          : 163,009  (99.6%)
  wasted compute       : 19.1 TFLOP
```

**99.6% waste.** And the scaling table:

```
  output len    total K/V work   vs 128 tokens
         128            16,320            1.0x
         512           163,584           10.0x
        2048         2,227,200          136.5x
```

16× the output, 136× the work, heading for quadratic. (Not the full 256×:
the fixed 64-token prompt contributes a large linear term at these lengths.)

---

## The idea

Autoregressive generation means each token depends on all previous ones. The
naive loop is the direct translation of that sentence:

```python
tokens = tokenizer(prompt, return_tensors="pt").input_ids

for _ in range(max_tokens):
    logits = model(tokens).logits          # the ENTIRE sequence, every time
    next_id = logits[:, -1].argmax(-1, keepdim=True)
    tokens = torch.cat([tokens, next_id], dim=-1)
```

Correct, and wasteful in a specific way. On step `n` the model computes keys and
values for all `n` tokens, but tokens `0..n-2` haven't changed since last step,
and neither have their K/V. Attention is causal: token 5's key never depends on
token 6. Recomputing it is pure waste.

Only `logits[:, -1]` is used. Everything else is discarded.

So step `n` costs O(n), and generating N tokens costs **O(N²)**.

### What to notice while implementing

- **`logits[:, -1]`**: the last position's prediction. Off-by-one here is the
  most common bug, and it produces plausible-but-wrong text rather than a crash.
- **`torch.no_grad()`**: no backward pass; gradients would waste memory.
- **`model.eval()`**: turns off dropout. Non-deterministic output otherwise.
- **`on_token()`**: call once per generated token so the benchmark harness can
  timestamp it. The test checks the count exactly; if it's wrong, every tok/s
  number in the course is off by the same factor and nothing else would catch it.

---

## Build it

**1. Implement `engine/model.py::load`.**

```python
def load(model_id=MODEL_ID, device=None, dtype=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or detect_device()
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype or torch.float32)
    model.to(device).eval()
    return model, AutoTokenizer.from_pretrained(model_id)
```

Use **float32 on MPS** to start. fp16 on Apple silicon has accuracy quirks that
will make the correctness test fail for reasons that have nothing to do with your
loop, fight one battle at a time.

**2. Implement `engine/generate.py::generate_naive`** using the sketch above.

**3. Prove it's right:**

```bash
uv run pytest tests/test_03_generation.py -v
```

On a fresh checkout these **skip** (`4 skipped`), they need the model
downloaded and `generate_naive` implemented. That is expected, not a failure.

Your greedy output must match HuggingFace's **exactly**. Greedy is deterministic,
so a mismatch is a bug, not noise. This test is the foundation for the rest of
the course, from Lecture 05 on, every faster version is checked against the same
reference, so you find out the moment speed costs you correctness.

**4. Measure: the actual point of the lecture:**

```bash
uv run python book/code/naive_bench.py
```

This generates at 128/256/512/1024 tokens and plots per-token latency against
position.

**Before running it, write your prediction in `notes/00-baseline/README.md`.**
Straight line? Curve? Flat? Commit to an answer.

**5. Also fill in `model_dims()`** in `engine/model.py` and check the real config
against the `ModelDims` guesses in `book/code/roofline.py`. If they differ, the
roofline numbers you computed in Lecture 02 need updating.

---

## What you should see

Per-token time climbing with position, roughly linearly.

If it looks **flat at short lengths**, that's not a contradiction: it's fixed
overhead (Python, kernel launches) dominating while sequences are short. The
quadratic term wins eventually. Note where the bend happens; that crossover point
is itself informative, and it's what Lecture 13 (CUDA graphs) attacks.

Save the plot. Lecture 05 overlays the cached version on it.

---

## Go deeper

- **Kiely §2.2** (p.46–49), LLM inference mechanics.
- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** §3.2.3, the
  causal masking that makes caching valid at all.
- **HuggingFace `generate()`**: `transformers/generation/utils.py`. Enormous,
  because it handles beam search, constraints, stopping criteria. Yours does one
  thing. Worth a look to see how much of a production API is edge cases.

---

## Check yourself

1. Your measured curve: linear, quadratic, or flat? Does it match your
   prediction? *(If not, write down why you were wrong. That's the valuable
   part.)*
2. The demo says 99.6% of K/V work is wasted. Your measured slowdown from 128 →
   1024 tokens is smaller than 64×. What else is the GPU spending time on?
3. Prefill was 512 ops:byte and decode 0.79. In your measurement, what fraction
   of wall time went to prefill? Compare to your Lecture 01 prediction.

---

## Next

**[04. Measuring](04-measuring.md)**: make sure the numbers you just took are
real. Short lecture, and it decides whether everything downstream is trustworthy.

Its one exercise: time a forward pass **with and without** `synchronize()`.
Seeing a fake 100× speedup yourself is worth more than being told it happens.

# 21 — JAX and XLA

**Build:** `jaxlm/model.py`, `jaxlm/decode.py` · **Test:** `tests/test_21_jax.py`
**Moves:** nothing directly — it changes how you *think* about the next two lectures
**Prereq:** [20 — Raw CUDA](20-raw-cuda.md)

---

## Why JAX is in this book

You've spent Part III controlling the machine directly — kernels, memory, threads.
JAX is the opposite philosophy, and seeing both is the point.

> **PyTorch:** you say *how*. Write the kernel, place the tensor, insert the
> collective.
> **JAX/XLA:** you say *what*, and annotate constraints. The compiler decides how.

That matters here for one specific reason: **sharding**. In Lecture 22 you'll do
tensor parallelism twice — once declaratively in JAX, where you annotate a layout
and XLA inserts the collectives, and once by hand in PyTorch with explicit
`all_reduce` calls.

Doing the declarative version first makes the manual version legible. You'll know
what the collectives are *for* before you write them, because you'll have seen a
compiler derive them from a layout.

It also generalizes: this is how TPU serving works, and how large-scale training
is increasingly expressed.

---

## The idea

### Pure functions and explicit state

JAX functions must be pure — no mutation, no hidden state. Parameters go in as
arguments:

```python
def forward(params, tokens, cache):
    ...
    return logits, new_cache        # a NEW cache, not a mutated one
```

This feels wasteful and isn't. XLA sees the whole dataflow, so it reuses buffers
in place when it can prove that's safe. **Functional at the source level, mutating
at the machine level.**

The relevant consequence for you: a KV cache in JAX is a value threaded through
the computation, not an object you append to. That's a genuinely different
formulation of Lecture 05, and expressing it twice is clarifying.

### `jit` compiles whole functions

```python
@jax.jit
def decode_step(params, token, cache):
    return forward(params, token, cache)
```

The first call traces the function into an XLA graph and compiles it — fusing
operations, allocating buffers, eliminating dead code. Later calls run the
compiled artifact.

Two things follow, and both should feel familiar from Lecture 13:

**Shapes are static.** A new input shape triggers recompilation. Same constraint
as CUDA graphs, same fix: bucket your shapes and pad.

**The first call is slow.** Compilation time. Warm up before measuring — the
Lecture 04 rule, again.

### `scan` for the decode loop

A Python loop over 512 decode steps unrolls into 512 copies of the graph.
Compilation takes forever. `lax.scan` expresses the loop *inside* the graph:

```python
def decode_body(carry, _):
    token, cache = carry
    logits, cache = forward(params, token, cache)
    next_token = jnp.argmax(logits[-1])
    return (next_token, cache), next_token

(final_token, final_cache), tokens = jax.lax.scan(
    decode_body, (first_token, cache), None, length=max_tokens
)
```

The **carry** is loop state — your token and KV cache. This is the same structure
as your PyTorch decode loop, made explicit enough for a compiler to reason about.

### Read the HLO

The genuinely instructive part:

```python
print(jax.jit(decode_step).lower(params, token, cache).compile()
      .as_text()[:4000])
```

You see the actual operations XLA chose — fusions, layout assignments, buffer
reuse. Compare against a `torch.compile` dump of the same model. **Two compilers,
same problem, different decisions.** Where they differ is where you learn what
compilation is actually doing.

---

## Build it

1. Implement Qwen3's forward pass in `jaxlm/model.py` — pure functions, params as
   a pytree. Load the same weights you've been using.
2. **Verify numerically against PyTorch** (`tests/test_21_jax.py`). Same weights,
   same input, same logits within fp32 tolerance. Do this before anything else;
   a silent transcription bug in RoPE or attention will waste hours later.
3. Implement `scan`-based decode with the KV cache as carry.
4. Dump and read the HLO. Find one fusion XLA performed that PyTorch eager
   wouldn't.
5. Benchmark against your PyTorch engine — **after warmup**.

**Fair-comparison note:** don't read too much into the headline number. Your JAX
version has no paging, no continuous batching, no prefix caching. It's a
single-sequence decode loop. The comparison worth making is *compiled JAX vs.
`torch.compile`*, not JAX vs. your whole engine.

---

## What you should see

**Comparable single-sequence performance.** Both compilers do similar work on this
shape of problem.

**Notably slow first call.** Compilation.

**Recompilation on shape change** — deliberately trigger it and watch the stall.
It's the same failure mode that bites `torch.compile` in production.

**Cleaner fusion in the HLO** than eager PyTorch, and a decode step that reads as
one fused region rather than a hundred kernel launches.

---

## Go deeper

- **[JAX: The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)**
  — read before you debug. Purity, PRNG keys, and shape-static-ness.
- **[`lax.scan` documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)**
  — the carry/output split takes a moment to click.
- **[XLA operation semantics](https://openxla.org/xla/operation_semantics)** —
  for reading the HLO.
- **[MaxText](https://github.com/AI-Hypercomputer/maxtext)** — production JAX LLM
  code; useful for seeing conventions.
- **Kiely §4.2.3** (p.104) — where compilers sit in the inference stack.

---

## Check yourself

1. A JAX cache is returned rather than mutated. Why isn't that a copy per step?
2. Why does a Python decode loop compile badly, and how does `scan` fix it?
3. What do static shapes have in common with CUDA graphs (Lecture 13)?
4. Name one fusion in your HLO and what it saved, in Lecture 02's terms.
5. Your JAX decode is slower than your engine. Why is that not a meaningful
   comparison as stated?

---

## Next

**[22 — Tensor parallelism](22-tensor-parallelism.md)** — the payoff: sharding
declared, then sharding by hand.

> **Needs 2+ GPUs** for the scaling curve. The sharding *arithmetic* is testable
> on one device (`pytest tests/test_22_tp.py -m "not cuda"`), so you can do the
> logic anywhere and rent only for the measurement.

The moment to look for: find the `all-reduce` **XLA inserted that you never
wrote**, then write it yourself in the same position.

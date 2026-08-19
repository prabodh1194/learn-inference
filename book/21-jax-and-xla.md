# 21. JAX and XLA

**Build:** `jaxlm/model.py`, `jaxlm/decode.py` · **Test:** `tests/test_21_jax.py`
**Moves:** nothing directly, it changes how you *think* about the next two lectures
**Prereq:** [20. Raw CUDA](20-raw-cuda.md)

---

## Why JAX is in this book

You've spent Part III controlling the machine directly, kernels, memory, threads.
JAX is the opposite philosophy, and seeing both is the point.

> **PyTorch:** you say *how*. Write the kernel, place the tensor, insert the
> collective.
> **JAX/XLA:** you say *what*, and annotate constraints. The compiler decides how.

Three terms to fix before they appear again. **XLA** is Google's compiler: it
takes the computation you described and produces GPU code, making the low-level
decisions that Part III taught you to make by hand. **Sharding** is splitting
one big array of numbers across several GPUs, each holding its slice. A
**collective** is an operation that needs the GPUs to cooperate, passing data
between themselves over their link, such as an all-reduce, which every GPU will
meet by name in Lecture 22, a sum spread over all of them so each ends with the
total.

Those definitions matter here for one specific reason: **sharding**. In Lecture
22 you'll do tensor parallelism (splitting a matmul across GPUs, each one
computing part of it) twice, once declaratively in JAX, where you annotate a
layout and XLA inserts the collectives, and once by hand in PyTorch with explicit
`all_reduce` calls.

Doing the declarative version first makes the manual version legible. You'll know
what the collectives are *for* before you write them, because you'll have seen a
compiler derive them from a layout.

It also generalizes: this is how TPU serving works (TPU, Google's own custom
AI chip, is JAX-first and sharded by construction), and how large-scale training
is increasingly expressed.

---

## The idea

### Pure functions and explicit state

The objective here is understanding why JAX refuses mutation: the rule is a
contract that buys the compiler the right to reuse buffers, and everything else
follows from that.

JAX functions must be pure, no mutation, no hidden state. Parameters go in as
arguments:

```python
def forward(params, tokens, cache):
    ...
    return logits, new_cache        # a NEW cache, not a mutated one
```

This feels wasteful and isn't. XLA sees the whole dataflow, so it reuses buffers
in place when it can prove that's safe. **Functional at the source level, mutating
at the machine level.** If the compiler can show that nothing reads a buffer
after one operation writes it, it lets the write happen over the old contents
instead of allocating fresh memory; you wrote "a new cache", the machine wrote
in place. Purity is a contract you offer so the compiler can be aggressive.

Picture the two levels, because "functional but mutating" sounds contradictory
until you see where each is true:

```
   what you WROTE                    what the machine RUNS

   cache ──► forward ──► cache'      ┌─────────────┐
   (two distinct values)             │  one buffer │◄── written over
                                     └─────────────┘    in place

   no mutation anywhere              nothing else reads the old
                                     contents, so overwriting is safe
```

**But there is a catch across `jit` boundaries, and it will cost you memory if
you miss it.** Inside one compiled function XLA proves this for itself. Between
calls it cannot: your Python still holds a reference to the old cache, so JAX
must assume you might use it and allocates a fresh buffer for the new one. At
Lecture 05's sizes that means carrying two copies of a multi-gigabyte cache.

The fix is to *promise* you won't touch the old one:

```python
@partial(jax.jit, donate_argnums=(2,))     # argument 2 (cache) may be reused
def decode_step(params, token, cache):
    return forward(params, token, cache)
```

`donate_argnums` marks an argument as dead on arrival — JAX is then free to
write the output over its buffer. Touch the donated value afterwards and you get
a loud error rather than silent corruption, which is the right trade.

The relevant consequence for you: a KV cache in JAX is a value threaded through
the computation, not an object you append to. That's a genuinely different
formulation of Lecture 05, and expressing it twice is clarifying.

### `jit` compiles whole functions

```python
@jax.jit
def decode_step(params, token, cache):
    return forward(params, token, cache)
```

The first call **traces** the function into an XLA graph and compiles it — the
just-in-time in JIT, compile on first use rather than in advance — fusing
operations (merging several ops into one kernel so intermediate values never
leave the chip), allocating buffers, eliminating dead code. A trace is a rehearsal
with fake placeholder values: JAX runs your Python function once, but instead
of computing, every operation writes down what it would have done, building the
graph. Later calls run the compiled artifact.

Two things follow, and both should feel familiar from Lecture 13:

**Shapes are static.** A new input shape triggers recompilation. Same constraint
as CUDA graphs, same fix: bucket your shapes and pad.

**The first call is slow.** Compilation time. Warm up before measuring, the
Lecture 04 rule, again.

### `scan` for the decode loop

Why a plain Python loop fails here is worth understanding, because it is the
same trap as Lecture 13's.

Tracing runs your Python **for real, once**, recording operations as it goes. So
a `for` loop doesn't become a loop in the graph — it becomes 512 *copies* of the
body, written out end to end:

```
   Python loop, traced          lax.scan, traced
   ┌──────────────┐             ┌──────────────┐
   │  step 1 ops  │             │   body ops   │◄─┐
   ├──────────────┤             └──────┬───────┘  │  one copy,
   │  step 2 ops  │                    └──────────┘  with a loop
   ├──────────────┤                                  marker
   │      ...     │  ×512
   ├──────────────┤             graph size: 1 body
   │ step 512 ops │             compile time: constant
   └──────────────┘
   graph size: 512 bodies
   compile time: minutes
```

One enormous graph, compiled once but slowly — and the compile time grows with
your token limit, which is absurd. `lax.scan` puts the loop *inside* the graph
instead:

```python
def decode_body(carry, _):
    token, cache = carry                              # unpack the state
    logits, cache = forward(params, token, cache)
    next_token = jnp.argmax(logits[-1])
    return (next_token, cache), next_token
#          └──── carry ─────┘  └── output ──┘

(final_token, final_cache), tokens = jax.lax.scan(
    decode_body,             # the body
    (first_token, cache),    # initial carry
    None,                    # xs: no per-step INPUT (we generate, not consume)
    length=max_tokens,       # ...so say how many steps explicitly
)
```

**The body returns two things, and the difference between them is the whole
API.** The `carry` is threaded to the next step; the output is *stacked* into an
array you get at the end:

```
              carry ────────► carry ────────► carry ────────► final_carry
   init  ──►  (tok,cache)     (tok,cache)     (tok,cache)
                  │               │               │
                  ▼               ▼               ▼
   outputs      tok_1           tok_2           tok_3     ──► stacked:
                                                              tokens[3]
```

So `final_cache` is the KV cache after the last step, and `tokens` is every
token generated, already assembled — you never append to a list.

The `None` third argument is where per-step *inputs* would go, if you had any
(`scan` over a batch of data, one row per step). Decode has none — each step's
input is the previous step's output — so it's `None`, and `length` tells `scan`
how many times to run.

This is the same structure as your PyTorch decode loop, made explicit enough for
a compiler to reason about.

### Read the HLO

The genuinely instructive part:

```python
print(jax.jit(decode_step).lower(params, token, cache).compile()
      .as_text()[:4000])
```

**HLO** is XLA's intermediate representation: the compiler's working draft of
your program, the graph of operations it settled on, in text form.
You see the actual operations XLA chose, fusions, layout assignments, buffer
reuse. Compare against a `torch.compile` dump of the same model. **Two compilers,
same problem, different decisions.** Where they differ is where you learn what
compilation is actually doing.

---

## Build it

1. Implement Qwen3's forward pass in `jaxlm/model.py`, pure functions, params as
   a pytree (JAX's name for a nested data structure, a dict of arrays; the
   compiler walks it like one object, and a new set of weights is just a new
   value). Load the same weights you've been using.
2. **Verify numerically against PyTorch** (`tests/test_21_jax.py`). Same weights,
   same input, same logits. Do this before anything else; a silent transcription
   bug in RoPE or attention will waste hours later.

    Be deliberate about dtype, or this step will fail for reasons that have
    nothing to do with your code. JAX defaults to fp32 for *its own literals and
    dtype promotion*, but arrays loaded from fp16 weights stay fp16, and matmul
    precision on GPU is a separate setting again
    (`jax.default_matmul_precision`). Your PyTorch engine has been running
    fp16. So either load both in the same dtype and compare tightly, or expect
    fp16-level disagreement (~1e-2) and set your tolerance accordingly. A
    mismatch here is usually dtype, not a bug in your attention.
3. Implement `scan`-based decode with the KV cache as carry.
4. Dump and read the HLO. Find one fusion XLA performed that PyTorch eager
   (its default mode, which runs each op immediately without compilation)
   wouldn't.
5. Benchmark against your PyTorch engine, **after warmup** — and **block before
   you stop the clock**. JAX dispatches asynchronously, exactly like CUDA
   (Lecture 04): calling `decode_step(...)` returns as soon as the work is
   *queued*, not when it is done. Time it without blocking and you will measure
   dispatch, get an implausibly fast number, and believe it:

    ```python
    out = decode_step(params, token, cache)   # returns immediately
    out[0].block_until_ready()                # THIS is where you stop the clock
    ```

    `block_until_ready()` is JAX's `torch.cuda.synchronize()`. The first call
    also pays compilation, so warm up first, then measure.

**Fair-comparison note:** don't read too much into the headline number. Your JAX
version has no paging, no continuous batching, no prefix caching. It's a
single-sequence decode loop. The comparison worth making is *compiled JAX vs.
`torch.compile`*, not JAX vs. your whole engine.

---

## What you should see

**Comparable single-sequence performance.** Both compilers do similar work on this
shape of problem.

**Notably slow first call.** Compilation.

**Recompilation on shape change**: deliberately trigger it and watch the stall.
It's the same failure mode that bites `torch.compile` in production.

**Cleaner fusion in the HLO** than eager PyTorch, and a decode step that reads as
one fused region rather than a hundred kernel launches (each a separate
CPU-to-GPU dispatch with its own overhead).

---

## Go deeper

- **[JAX: The Sharp Bits](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)**:
  read before you debug. Purity, PRNG keys, and shape-static-ness.
- **[`lax.scan` documentation](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html)**
 : the carry/output split takes a moment to click.
- **[XLA operation semantics](https://openxla.org/xla/operation_semantics)**:   for reading the HLO.
- **[MaxText](https://github.com/AI-Hypercomputer/maxtext)**: production JAX LLM
  code; useful for seeing conventions.
- **Kiely §4.2.3** (p.104), where compilers sit in the inference stack.

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

**[21b. Model compilation](21b-model-compilation.md)**: the same compiler idea
on the serving side — torch.compile and TensorRT-LLM. Then
**[22. Tensor parallelism](22-tensor-parallelism.md)**: the payoff: sharding
declared, then sharding by hand.

> **Needs 2+ GPUs** for the scaling curve. The sharding *arithmetic* is testable
> on one device (`pytest tests/test_22_tp.py -m "not cuda"`), so you can do the
> logic anywhere and rent only for the measurement.

The moment to look for: find the `all-reduce` **XLA inserted that you never
wrote**, then write it yourself in the same position.

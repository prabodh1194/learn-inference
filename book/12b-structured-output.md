# 12b. Structured output and adapters

**Build:** a logit-processor hook in `engine/sampling.py` · **Test:** `tests/test_12b_structured.py`
**Moves:** output *validity*, and reveals a scheduling cost most people miss
**Prereq:** [12. Speculative decoding](12-speculative-decoding.md)

---

## Why this lecture exists

Everything so far has optimized *how fast* tokens come out. Production serving
also cares about *what* comes out, and about serving many models from one engine.

Three features you'll meet immediately in any real deployment, all first-class in
vLLM, none of them free:

- **Structured output**: force valid JSON, or a schema, or a grammar
- **Tool calling**: the dominant use case for LLM inference in 2026
- **LoRA adapters**: serve many fine-tunes from one set of base weights

They're grouped here because each is a *constraint on the engine* rather than a
speed optimization, and each interacts with machinery you've already built.

---

## Structured output

An agent calling a tool needs valid JSON. Asking politely in the prompt gets you
valid JSON *most* of the time, and "most" is a production incident.

The mechanism is simple and elegant: **mask the logits.**

Each step is a choice among the 151,936 vocabulary tokens, ranked by score. The
scores are the **logits** from Lecture 06, one unnormalized score per token.
Masking means crossing the illegal choices off the list before the pick happens:
a state machine tracks where you are in the grammar and computes which tokens
could legally come next, and everything else is set to `-inf`, minus infinity,
before sampling:

```python
allowed = grammar_fsm.allowed_tokens(state)     # e.g. only '"' after '{'
logits[~allowed] = float("-inf")
token = sample(logits, params)
state = grammar_fsm.advance(state, token)
```

Why `-inf` works: the next step turns scores into probabilities (softmax,
Lecture 06), and a score of `-inf` becomes exactly 0%, no rounding, no
exceptions. Invalid output becomes **impossible**, not unlikely. That's a much
stronger guarantee than prompting, and it composes with any sampling parameters.

A tiny grammar in action (a schema like `{ "city": "..." }`):

```
after '{'           legal: only '"'                (a key must start)
after '{"ci'        legal: any letter, or '"'      (still the key)
after '{"city"'     legal: only ':'                (forced)
after '{"city":'    legal: only '"'                (a string value)
after '{"city":"S"  legal: any letter, or '"'      (still the value)
```

This is your Lecture 06 sampler with one hook added, which is why it lands here.

### Where it gets expensive

The mask must be computed **per step, per sequence**: each step's answer is a
list of 151,936 legal/illegal flags, one per vocabulary token, and the list
changes every time the grammar position advances. Multiply by the number of
sequences in the batch, every step. Naively that's a huge amount of CPU work on
the critical path, and it lands squarely in the gap Lecture 24 warned you
about: CPU work that stalls the GPU.

Real implementations attack this with:

- **Precompiled FSMs.** Compile the schema once, cache it, reuse across requests
  with the same schema, which is most of them, since an app has a handful of
  tools.
- **Token-level tries.** Precompute vocabulary-to-grammar transitions rather than
  testing tokens one at a time.
- **Jump-ahead / fast-forward.** When the grammar *forces* the next tokens (after
  `{"na` in a schema, `me":` is the only legal continuation), emit them **without
  running the model at all.** Free tokens.

That last one is worth pausing on: it's the same insight as speculative decoding,
tokens you can predict with certainty don't need a forward pass, arrived at from
a completely different direction.

The libraries: **XGrammar** (vLLM's default), **Outlines**, **llguidance**.

### The interaction people miss

Grammar state is **per sequence**, and it must survive everything your scheduler
does:

- **Preemption** (L09), a swapped-out sequence must restore its FSM state
- **Prefix caching** (L10), two requests sharing a prefix may have *different*
  schemas, so grammar state cannot be shared along with the blocks
- **Speculative decoding** (L12), drafted tokens must be checked against the
  grammar, not just against the target model

vLLM's compatibility matrix documents exactly which of these combinations work
together. Read it as a map of where the hard problems are.

---

## Tool calling

Tool calling is structured output plus a protocol. The model emits a constrained
JSON block naming a function and its arguments; your server parses it out of the
stream and hands it back to the client.

The inference-side complications, which look small and aren't:

**Parsing streams.** You're emitting SSE token by token (server-sent events,
the standard way to stream a response over one long HTTP connection), but a
tool call is only valid once complete. You either buffer (killing streaming for
that segment) or emit incremental parse events. Every serving framework has a
per-model **tool-call parser** for exactly this reason.

**Model-specific formats.** Qwen, Llama, and Mistral all emit tool calls
differently. The parser is per model family, and getting it wrong yields silently
malformed calls.

**Interaction with speculation.** The [field notes](field-notes.md) record a real
case of this: an operator found tool calling was **inaccurate when MTP
(multi-token prediction, a speculative-decoding scheme) was enabled**, a genuine
bug in the parser interaction, which they fixed by cherry-picking a PR.
Features that are individually correct can be jointly wrong.

---

## LoRA adapters

You've fine-tuned twelve variants of one base model. Loading twelve full copies
wastes VRAM on twelve nearly identical weight sets.

LoRA stores each fine-tune as a low-rank update instead of a fresh weight set:
`W + BA`, where `B` and `A` are two skinny matrices whose product has the same
shape as the original `W`. "Low-rank" is the mathematical name for that trick,
rebuilding a big matrix from two small ones: with rank `r`, the pair stores
`2 · r · hidden` numbers against the `hidden²` of the full matrix. For a square
weight matrix, as the K and V projections are (hidden → hidden), a rank-8
adapter stores 16,384 numbers against 1,048,576, a factor of
1024 / (2 × 8) = 64 fewer per weight matrix. So keep
**one** copy of the base weights and swap in adapters per request:

```
base weights (840 MiB)  shared by every request
  + adapter A (a few MiB)  -> request 1
  + adapter B (a few MiB)  -> request 2
```

**Multi-LoRA batching** takes this further: requests using *different* adapters in
the *same* batch, with a fused kernel (one chip program that applies the right
adapter to each sequence in a single pass) applying the right adapter per
sequence. That preserves the continuous batching you built in Lecture 08,
otherwise you'd have to group requests by adapter and lose the scheduling
freedom entirely.

The cost is real but modest: extra kernels per layer, and adapter weights
competing with the KV cache for memory (Lecture 09's budget again).

**And a correctness note that connects to Lecture 10:** adapter identity *must* be
part of the prefix cache's block hash. The same tokens under a different adapter
produce different K/V. This is precisely why vLLM's `kv_cache_utils.py` includes
LoRA id in the hash: the scar tissue Lecture 14 pointed you at.

---

## Build it

This lecture is deliberately lighter on implementation: the ideas matter more
than a toy version.

1. Add a **logit-processor hook** to `engine/sampling.py`: a callable that can
   mask logits before sampling. That's the extension point everything above uses.
2. Implement a **minimal JSON grammar**, enough to force balanced braces and
   quoted keys. Not a full schema engine; enough to feel the mechanism.
3. `uv run pytest tests/test_12b_structured.py -v`
4. **Measure the overhead.** Generate with and without the mask, and report the
   per-step CPU cost. Then ask: at batch 32, is this on the critical path?
5. Read vLLM's compatibility matrix and note which features *cannot* be combined.
   That table is a map of unsolved problems.

---

## Go deeper

- **[XGrammar](https://arxiv.org/abs/2411.15100)**: vLLM's default backend;
  explains the token-trie and jump-ahead optimizations.
- **[Outlines](https://arxiv.org/abs/2307.09702)** (Willard & Louf), regex and
  grammar-guided generation via FSM indexing. The clearest statement of the core
  idea.
- **[S-LoRA](https://arxiv.org/abs/2311.03285)**: serving thousands of LoRA
  adapters concurrently; unified paging for adapters and KV cache.
- **[Punica](https://arxiv.org/abs/2310.18547)**: the multi-LoRA batching kernel.
- **[vLLM feature compatibility matrix](https://docs.vllm.ai/en/latest/features/)**
 , which combinations work. Genuinely useful as a research map.
- **Gordić, *Inside vLLM***, has a guided-decoding (FSM) section. Another reason
  Lecture 14 is where it belongs.

---

## Check yourself

1. Why does logit masking give a *guarantee* where prompting gives a probability?
2. Grammar masking is CPU work per step per sequence. Why is that dangerous
   specifically, given Lecture 24?
3. How is grammar jump-ahead the same idea as speculative decoding?
4. Two requests share a 400-token prefix but use different LoRA adapters. Can they
   share KV blocks? Why not?
5. Tool calling broke when speculation was enabled. What class of bug is that, and
   what does it suggest about testing feature combinations?

---

## Next

**[13. CUDA graphs](13-cuda-graphs.md)**: back to raw speed, and the first
lecture where the GPU *isn't* the bottleneck.

> **NVIDIA GPU required**, no equivalent exists on MPS, and the tests are
> marked `cuda` so they skip cleanly on a laptop. If you're still local, read it
> and move to L14; come back when you rent.

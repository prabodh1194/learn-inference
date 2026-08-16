# 12b. Structured output and adapters

**Build:** a logit-processor hook in `engine/sampling.py` · **Test:** `tests/test_12b_structured.py`
**Moves:** output *validity*, and reveals a scheduling cost most people miss
**Prereq:** [12. Speculative decoding](12-speculative-decoding.md)

---

## Why this lecture exists

Everything so far made the engine *fast*. This lecture makes it *usable* — a
fast engine that emits invalid output, can't call a tool, and only serves one
model is not something you can put in production.

Set the scene with what a real client actually sends. It asks for a JSON object
(`response_format: {"type": "json_object"}`), not prose — because the next stage
of its pipeline throws on anything else. Then it asks for a *tool call*: the
model doesn't answer, it acts, and the engine must emit a function invocation
the client can parse and run. And the app has twelve fine-tuned variants of your
base model it wants served without loading twelve copies.

Three requirements, all from production, none of them about speed:

- **Structured output**: force valid JSON, a schema, or a grammar. "Mostly
  valid" is a production incident.
- **Tool calling**: the dominant use case for LLM inference in 2026 — and it is
  structured output plus a protocol.
- **LoRA adapters**: serve many fine-tunes from one set of base weights.

They belong in one lecture because they are the same move from three angles:
the engine **constrains what it emits**. Structured output masks the logits so
only legal tokens survive; tool calling constrains to the tool's schema and
parses the stream; LoRA constrains *which weights* apply to the request. Each
reuses machinery you already built — the sampler (L06), the KV cache (L05), the
scheduler (L08), the prefix cache (L10) — and each costs the engine something
non-obvious that only shows up once you wire it in. None of them makes a token
arrive faster; all of them are what turn an engine into a service.

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
the critical path, and it lands squarely in the gap Lecture 13 warned you
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

The libraries: **XGrammar** (vLLM's default), **Outlines**, **llguidance**,
and **llama.cpp's GBNF** (a context-free-grammar format; the "response_format =
JSON object" sugar is convenience syntax over the same masking machinery).

### The quality cost of masking

Masking guarantees *validity*, not *quality*, and it has a subtle failure mode
worth knowing: it can force a legal-but-unlikely *tokenization*.

The canonical case is the URL: the model wants to emit `http` + `://` (two
slashes, no space), but the grammar put the colon on one edge and the slash on
another, so the mask admits only `http` + `:` + `//`. Both are the same string;
one path has near-zero probability mass under the model. Forcing it *changes
what the model would have said*. **Token healing** fixes it: when the mask
picks a low-mass continuation, roll back one token and require the next to start
with the desired prefix — the mirror image of jump-ahead (which emits forced
tokens; healing un-forces a bad one). It's off by default because a rollback
means a recomputed forward pass; that cost is why it's a heuristic, not a given.

There is also a ceiling on what masking *can* express. A finite state machine
cannot count: schemas with unbounded nesting (`{"a":{"a":{...}}}`) need a
**stack**, so real backends compile JSON schemas to a pushdown automaton, not
an FSA — and that stack is part of the per-sequence state that must survive
preemption (next section). Constraints like "all keys unique" or "every `$ref`
is defined before use" are beyond even context-free: no masking scheme can
enforce them at all. The schema is a contract; the grammar only guarantees
*syntax*, and required keys mean distinct FSM states, not just balanced braces.

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
malformed calls. The industry standard for the *transport* is **MCP** (Model
Context Protocol), but the per-model JSON-schema format still varies.

**Tool definitions are prompt tokens.** Every tool's name, description, and
input schema is serialized into the prompt, so it costs KV cache — a large tool
set is prefill traffic and permanent context. This is why real agents ship a
name-only menu and **deferred schemas**: load a tool's full schema only when the
model actually reaches for it.

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
   quoted keys. Not a full schema engine; enough to feel the mechanism. Note
   where it *can't* help: duplicate keys and missing required keys are both
   valid JSON syntactically, so a syntax-only mask admits them.
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
   specifically, given Lecture 13?
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

# 26b. SGLang internals

**Build:** nothing · **Test:** none · **Moves:** your ability to read the other production engine
**Prereq:** [26. Versus vLLM](26-versus-vllm.md), [13. CUDA graphs](13-cuda-graphs.md), [22. Tensor parallelism](22-tensor-parallelism.md)

---

## The problem

You know vLLM. You've read its source (Lecture 14). But half your incidents
happened in SGLang — `enable_thinking` no-op, model runner subprocess isolation,
TP=2 capture crash, `SGLANG_EXTERNAL_MODEL_PACKAGE` deprecation. You need to
read SGLang the way you read vLLM: not to learn what continuous batching is,
but to see what a different team did with the same ideas after different
constraints.

---

## The idea

SGLang (Scalable Generative Language) started from a different premise:
**CUDA graphs by default, not as an optimisation.** vLLM added graphs later;
SGLang built the engine *around* graph capture. That choice cascades into
everything: the scheduler, the memory pool, the model runner, the template
mechanics.

Key architectural differences from vLLM:

| Aspect | vLLM | SGLang |
|--------|------|--------|
| Graph capture | Opt-in (`enforce_eager=False`) | Default path |
| Prefix caching | Block-level hash (L10) | **RadixAttention** — tree-structured prefix tree |
| Scheduler | Chunked prefill + continuous batching | **DAG scheduler** — RadixCache as prefix tree |
| Model runner | In-process (V0) / subprocess (V1) | Always subprocess |
| Template handling | Jinja via tokenizer | Strips template prefix from response |
| TP + graphs | Fragile | TP=1 recommended, DP for throughput |

---

## The method

### 1. RadixAttention — prefix reuse as a radix tree

vLLM's prefix cache (L10) hashes the *parent block* and stores blocks in a
flat LRU. SGLang builds a **radix tree** (trie) over token sequences:

```
                    [root]
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
           "The"    "A"     "Hello"
              │        │        │
              ▼        ▼        ▼
           "quick"   "cat"   "world"
              │
              ▼
           "brown"
```

Each node = a prefix. Reference count = how many active sequences share this
prefix. When a sequence ends, walk up decrementing counts; evict zero-count
subtrees. This gives **exact prefix sharing** — no hash collisions, no block
granularity waste. A 100-token shared prefix reuses 100 tokens of KV, not
"the largest matching block."

**Source to read:** `sglang/srt/managers/scheduler.py` → `RadixCache` class.
The `match_prefix` method walks the tree; `insert_prefix` builds it.

### 2. CUDA graphs by default — the capture architecture

vLLM: eager → optional graph capture. SGLang: **graphs are the happy path**.

The model runner captures graphs for each batch size it sees (configurable
`max_capture_batch_size`). On first request of a new batch size: capture →
store → replay. Subsequent requests of that size replay the graph.

**Why this matters for your incidents:**
- I9 (indexed assignment): boolean mask breaks capture → crash *during capture*
- I12 (TP=2): cross-device collectives inside captured region → capture fails
- The engine assumes *your model code is capturable*. If it's not, you hit I9.

**Source to read:** `sglang/srt/model_executor/model_runner.py` →
`_capture_model` and `forward` — the capture/replay loop.

### 3. The model runner subprocess + `SGLANG_EXTERNAL_MODEL_PACKAGE`

SGLang *always* runs the model in a subprocess (unlike vLLM V0). The parent
process (API server) communicates via `multiprocessing` queues. The child
(model runner):
- Has its own interpreter, `sys.path`, CUDA context
- Does not inherit `PYTHONPATH` from parent
- Loads models through its own import path

**Your incidents I5, I13:** `SGLANG_EXTERNAL_MODEL_PACKAGE` was meant to be the
extension point — point at a package, runner imports it. But:
- The env var doesn't propagate to spawn'ed child (I5)
- Even when it did, the runner's import machinery had bugs (I13)

**Source to read:** `sglang/srt/managers/model_runner.py` → `ModelRunner`
process startup, `_load_model` method.

### 4. Chat template mechanics — `enable_thinking` and prefix stripping

SGLang applies the chat template *in the model runner*, then **strips the
template prefix** from the generated response before returning to the API
server. This is intentional — the client shouldn't see the template scaffolding.

But: Qwen3's template *unconditionally* emits `think`. The `enable_thinking`
kwarg only works if the template author wrote `{% if enable_thinking %}`. Qwen3
didn't. So:
- Template emits `think` regardless
- SGLang strips the prefix → opening `think` disappears
- Output looks like a malformed thinking block (I2)

**Source to read:** `sglang/srt/conversation.py` → `Conversation` class,
`get_prompt` and the response post-processing in `generate`.

### 5. TP on B200 — why DP wins

Same as Lecture 22: TP is a latency optimisation (split one token across GPUs).
It inserts an all-reduce per layer. On B200 (183 GB VRAM), the model fits on
one card. For throughput-bound work:

| Strategy | Throughput | Latency | Capture |
|----------|------------|---------|---------|
| TP=2 | 1× (collective overhead) | 2× better (theory) | Cross-device = fragile |
| DP (2× single-GPU) | 2× | 1× | Each captures independently |

SGLang's TP=2 capture crash (I12) is the same root cause: cross-device ops in
a captured graph. The fix isn't "make TP capture work" — it's "use DP for
throughput."

**Source to read:** `sglang/srt/model_executor/parallel_utils.py` → tensor
parallel communication; `sglang/srt/managers/scheduler.py` → data parallel
dispatch.

---

## Build it

No code — source reading exercise. Clone SGLang at the version you're
deploying (e.g., `git checkout v0.5.13`), then read in this order:

1. **Entry points**: `sglang/srt/server.py` (API server), `sglang/srt/managers/scheduler.py` (scheduler + RadixCache)
2. **Model runner**: `sglang/srt/managers/model_runner.py` (subprocess, capture, model loading)
3. **RadixAttention**: `sglang/srt/managers/scheduler.py` → `RadixCache` class
4. **CUDA graph capture**: `sglang/srt/model_executor/model_runner.py` → `_capture_model`, `forward`
5. **Template handling**: `sglang/srt/conversation.py` → `Conversation.get_prompt`, response stripping
6. **Custom model loading**: `sglang/srt/model_executor/model_runner.py` → `_load_model`, `SGLANG_EXTERNAL_MODEL_PACKAGE` handling
7. **TP / DP**: `sglang/srt/model_executor/parallel_utils.py`, scheduler's `DataParallelDispatcher`

---

## What you should see

- RadixCache is a trie, not a flat LRU — exact prefix sharing, no block granularity loss
- Graph capture happens on first request per batch size; replay thereafter
- Model runner is a `spawn` subprocess with isolated `sys.path`
- Template prefix stripped from response before client sees it
- `enable_thinking` only works if template has the branch
- TP>1 fragile on capture; DP is the throughput path

---

## Go deeper

- [Production troubleshooting](troubleshooting.md#pattern-1-the-knob-that-does-nothing) — I2, I3, I5, I12, I13
- [13. CUDA graphs](13-cuda-graphs.md) — the constraints SGLang defaults to
- [22. Tensor parallelism](22-tensor-parallelism.md) — why TP is latency, not throughput
- SGLang paper: *SGLang: Structured Generation Language for Efficient LLM Serving* (Zheng et al., 2024)
- vLLM vs SGLang architecture comparison: `docs/source/benchmark/vllm_comparison.md` in SGLang repo

---

## Check yourself

1. Why does RadixAttention reuse more KV than vLLM's block-level prefix cache?
2. A custom model has `residual[:, mask] = 0.0`. It works in vLLM eager mode. Why does it crash in SGLang?
3. You set `SGLANG_EXTERNAL_MODEL_PACKAGE=my_models`. The model runner still can't import. Why?
4. Qwen3 emits `think` even with `enable_thinking=False`. Where are the two bugs?
5. B200, 183 GB VRAM, model fits on one GPU. TP=2 or DP for throughput? Why?

---

## Next

[27. Routing and disaggregation](27-routing-and-disaggregation.md) — now that you know both engines, how to route between them.
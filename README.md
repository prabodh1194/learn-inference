# Inference Zero-to-Hero

Building an LLM inference engine from scratch — Karpathy-style. Build the naive
thing, measure it, find the bottleneck, fix it, measure again, then read how the
pros did it.

The premise: you don't learn what a KV cache is by reading about it. You write
`generate()` without one, watch per-token latency climb, then add it and watch
the curve go flat.

**Model throughout:** `Qwen3-0.6B` — small enough to iterate on a laptop, real
enough to be non-trivial (GQA, RoPE, RMSNorm, SwiGLU). Same model start to finish
so every benchmark is comparable.

---

## The loop

```
build naive → measure → find the bottleneck → fix it → measure again → read the pros
```

The measurement is not optional. It's what converts trivia into intuition. Every
milestone below names a number that has to move.

---

## Progress

### Phase 0 — Baseline and instrumentation
> Build the ruler. Predict a bottleneck on paper before measuring it.

- [ ] **M0.1** Load Qwen3-0.6B, naive `generate()` — no KV cache, O(N²) on purpose
- [x] **M0.2** `bench/` harness — TTFT, TPOT, tok/s, p50/p90/p99, peak memory
- [ ] **M0.3** Plot time-per-token vs. position → the quadratic pain
- [x] **M0.4** Roofline + arithmetic intensity by hand ✅ *reproduces book's 62 ops:byte*

**Gate:** state, with your own numbers, why decode is memory-bound and prefill is compute-bound.

### Phase 1 — Build the engine (PyTorch)
> The heart of it. Starts on the laptop; rent a GPU from M1.5.

- [ ] **M1.1** KV cache → the curve flattens
- [ ] **M1.2** Sampling: temperature, top-k, top-p, repetition penalty
- [ ] **M1.3** Static batching → measure throughput gain *and* padding waste
- [ ] **M1.4** Continuous batching → split into scheduler + model runner
- [ ] **M1.5** Paged KV cache → measure max concurrent sequences before OOM
- [ ] **M1.6** Prefix caching → hash blocks, refcount, LRU evict
- [ ] **M1.7** Chunked prefill → judge on p99, not the mean
- [ ] **M1.8** Speculative decoding (n-gram) → report acceptance rate
- [ ] **M1.9** CUDA graphs + `torch.compile` → kill launch overhead

**🎯 Capstone:** read [Gordić's *Inside vLLM*](https://www.aleksagordic.com/blog/vllm),
then [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) (~1,200 lines).
Deliberately *after* building — you compare engineering judgment, not learn vocabulary.

**Gate:** you have a working engine and can read vLLM's source without it being magic.

### Phase 2 — Kernels (Triton → CUDA)
> Where "why is it fast" lives. Needs an NVIDIA GPU.

- [ ] **M2.1** Profile first (Nsight, `torch.profiler`) — rank kernels by time
- [ ] **M2.2** Triton warmup: vector add → softmax → RMSNorm
- [ ] **M2.3** FlashAttention-style tiled attention → re-plot the roofline
- [ ] **M2.4** Paged attention kernel (reads through the block table)
- [ ] **M2.5** Quantization — measure memory, throughput, **and quality**
- [ ] **M2.6** Rewrite one kernel in raw CUDA C

### Phase 3 — JAX and parallelism
> JAX makes sharding *declarative*, which makes vLLM's manual NCCL legible.

- [ ] **M3.1** Qwen3 forward pass in JAX, `jit`-ed
- [ ] **M3.2** `lax.scan` decode loop, KV cache as carry → inspect the HLO
- [ ] **M3.3** Tensor parallelism via `NamedSharding` → find where scaling bends
- [ ] **M3.4** Port TP back to PyTorch by hand with NCCL collectives
- [ ] **M3.5** MoE and expert parallelism

### Phase 4 — Production serving
> Where inference stops being kernels and becomes systems.

- [ ] **M4.1** OpenAI-compatible HTTP + SSE streaming
- [ ] **M4.2** Load testing: Poisson arrivals, concurrency sweep → find the knee
- [ ] **M4.3** Head-to-head vs. real vLLM/SGLang → **expect to lose**, then explain the gap
- [ ] **M4.4** Cache-aware routing
- [ ] **M4.5** Disaggregated prefill/decode
- [ ] **M4.6** Autoscaling + cost per million tokens

### Phase 5 — Contribute
- [ ] vLLM test suite green on a fork
- [ ] Land a PR (`good first issue` on vLLM/SGLang)
- [ ] Reproduce a recent inference paper against your own engine

---

## Layout

```
engine/     Phase 1 — the PyTorch engine, grown milestone by milestone
kernels/    Phase 2 — Triton, then raw CUDA
jaxlm/      Phase 3 — JAX model, scan decode, sharding
serve/      Phase 4 — HTTP, routing, disaggregation
bench/      ⚠️ built FIRST — used by every phase
notes/      the learning record: prose + runnable code
```

## Setup

```bash
uv venv --python 3.12
uv sync
uv run python notes/00-baseline/m04_roofline.py   # runs today, no GPU needed
```

Phase-specific extras (install on the rented box, not the laptop):

```bash
uv sync --extra kernels   # Triton — Linux/NVIDIA only
uv sync --extra jax
uv sync --extra serve
```

## Hardware

| Phase | Where |
|---|---|
| 0, M1.1–M1.4 | laptop — correctness and scheduling logic are CPU/MPS-fine |
| M1.5 onward | rented NVIDIA — paged attention and CUDA graphs need real CUDA |
| Phase 2 | rented, every session — Nsight requires NVIDIA |
| M3.3 | rented multi-GPU (2–4×) |
| Phase 4 | rented, longer blocks — load testing needs sustained runtime |

Read and write code locally, batch the GPU work, always stop the pod.

## Reference

- **[*Inference Engineering*](https://www.baseten.co/resources/book/inference-engineering/)** (Kiely, Baseten 2026) — the map. Read sections just-in-time; each milestone names its section.
- **[Inside vLLM](https://www.aleksagordic.com/blog/vllm)** (Gordić) — the Phase 1 capstone, read *after* building.
- **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** — ~1,200 readable lines; the nanoGPT of inference.
- **[vLLM](https://github.com/vllm-project/vllm)** / **[SGLang](https://github.com/sgl-project/sglang)** — the real thing.

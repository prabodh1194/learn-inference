# Inference Zero-to-Hero

*Building an LLM inference engine from scratch, measuring everything.*

You build KV caching, continuous batching, paged attention, prefix caching and
speculative decoding, then read vLLM and understand why it's built the way it is.

[Start reading :material-arrow-right:](00-intro.md){ .md-button .md-button--primary }
[The repo :fontawesome-brands-github:](https://github.com/prabodh1194/learn-inference){ .md-button }

!!! quote "The method"
    **build the naive thing → measure it → find the bottleneck → fix it → measure again**

    You write a slow generation loop on purpose, plot how badly it scales, and
    only *then* build the KV cache. Reading "the KV cache avoids recomputation"
    teaches you a sentence; watching your own per-token latency climb and then
    flatten teaches you the thing.

---

## How to read this

Each lecture gives you four things:

| | |
|---|---|
| **the text** | the idea, derived, 5–10 minutes |
| **a demo** | `uv run python book/code/NN_*.py`, shows the phenomenon |
| **a build** | you implement it in `engine/` |
| **a test** | `pytest tests/test_NN_*.py`, proves it's right |

And a **number that must move**. Record it in `notes/`.

**Run the demo before reading "The idea."** Seeing the waste as a number first is
what makes the fix feel inevitable rather than arbitrary.

!!! warning "Don't skip the predictions"
    Several lectures ask you to guess before measuring. Being wrong in writing is
    how the intuition forms; skipping to the answer feels efficient and teaches
    much less.

---

## Three ways in

<div class="grid cards" markdown>

-   :material-numeric-1-box: **Follow the path**

    ---

    Numbered steps 1–53, setup through vLLM, training systems, and GRPO/R1.

    [START-HERE](https://github.com/prabodh1194/learn-inference/blob/main/START-HERE.md)

-   :material-book-open-variant: **Read straight through**

    ---

    Intro, then Part I. Demos run with no GPU.

    [00. Introduction](00-intro.md)

-   :material-help-circle: **Chase a confusion**

    ---

    Worked answers to real questions, several of which found book errors.

    [Q&A](qa.md)

</div>

---

## Field notes

[**field-notes.md**](field-notes.md) collects what practitioners report, real
magnitudes from real deployments, and the places where an optimization
disappointed someone. Books give you the mechanism; these give you the scale.

Use them to sanity-check your own results, and add your own entries as you go.

---

## Companion texts

This book indexes two excellent sources rather than duplicating them:

- **Philip Kiely, *Inference Engineering*** (Baseten, 2026): the breadth-first
  survey. Lectures cite it by section; read those sections when pointed at them.
- **Aleksa Gordić, [*Inside vLLM*](https://www.aleksagordic.com/blog/vllm)**, a
  top-down read of vLLM V1. **Save it for Lecture 14.** Read before you've built
  a scheduler and it teaches vocabulary; read after and it teaches judgment.

---

## Status

**All 62 lectures are written** across seven parts — from inference foundations
and building a vLLM-style engine (Parts I–V) to pre-training systems (scaling laws, 5D parallelism, FP8, Muon) and
post-training alignment with SFT, LoRA, Rejection Sampling, PPO, DPO, GRPO/DAPO/GSPO, On-Policy Distillation, AReaL, and Model Merging
(Parts VI–VII).

The **code** is a different matter, and deliberately so: `engine/`, `kernels/`,
`jaxlm/`, `serve/`, and `train/` are stubs for *you* to fill in. The demos in
`book/code/` run today; the tests in `tests/` fail until you build each lecture.
That failure list is the course.

Run `uv run python scripts/progress.py` to see where you are.

---

## Table of contents

- **Part I — Foundations:** [00 Intro](00-intro.md) · [01 The two phases](01-the-two-phases.md) · [02 Arithmetic intensity](02-arithmetic-intensity.md) · [03 Naive generation](03-naive-generation.md) · [04 Measuring](04-measuring.md)
- **Part II — The Engine (PyTorch):** [05 KV cache](05-kv-cache.md) · [06 Sampling](06-sampling.md) · [06b Search-based decoding](06b-search-based-decoding.md) · [07 Static batching](07-static-batching.md) · [08 Continuous batching](08-continuous-batching.md) · [09 Paged attention](09-paged-attention.md) · [10 Prefix caching](10-prefix-caching.md) · [11 Chunked prefill](11-chunked-prefill.md) · [12 Speculative decoding](12-speculative-decoding.md) · [12b Structured output](12b-structured-output.md) · [13 CUDA graphs](13-cuda-graphs.md) · [14 Reading vLLM](14-reading-vllm.md)
- **Part IIb — h3.c (Metal):** [14b Reading h3.c](14b-reading-h3.md) · [14c Weight residency](14c-weight-residency.md) · [14d Fusion](14d-fusion.md) · [14e Activation lifetimes](14e-activation-lifetimes.md) · [14f CPU/GPU overlap](14f-cpu-gpu-overlap.md) · [14g Quantization with receipts](14g-quantization.md) · [14h When optimization changes outputs](14h-aggressive-optimization.md)
- **Part III — Kernels (Triton & CUDA):** [15 Profiling](15-profiling.md) · [16 Triton basics](16-triton-basics.md) · [17 FlashAttention](17-flash-attention.md) · [18 Paged attention kernel](18-paged-attention-kernel.md) · [19 Quantization](19-quantization.md) · [19b FP8](19b-fp8.md) · [19c GGUF](19c-gguf.md) · [20 Raw CUDA C](20-raw-cuda.md)
- **Part IV — Parallelism (JAX & NCCL):** [21 JAX & XLA](21-jax-and-xla.md) · [21b Model compilation](21b-model-compilation.md) · [22 Tensor parallelism](22-tensor-parallelism.md) · [23 MoE & expert parallelism](23-moe-and-expert-parallelism.md)
- **Part V — Production:** [24 Serving](24-serving.md) · [24b Serving agents](24b-serving-agents.md) · [25 Load testing](25-load-testing.md) · [25b Deployment environments](25b-deployment-environments.md) · [26 Versus vLLM](26-versus-vllm.md) · [26b SGLang internals](26b-sglang-internals.md) · [27 Routing & disaggregation](27-routing-and-disaggregation.md) · [28 Autoscaling & cost](28-autoscaling-and-cost.md) · [28b Reasoning & test-time compute](28b-reasoning-and-test-time-compute.md) · [29 Contributing](29-contributing.md)
- **Part VI — Training Systems:** [30 The training ledger ($6ND$ & $16\Psi$)](30-training-arithmetic.md) · [30b Scaling laws, WSD & mid-training](30b-scaling-laws-and-schedules.md) · [31 Activations & checkpointing](31-activation-memory.md) · [32 Data curation, MinHash & packing](32-data-and-packing.md) · [33 Data parallelism, ZeRO & FSDP](33-zeRO-and-fsdp.md) · [34 TP + SP, Pipeline schedules & 3D](34-3d-parallelism-and-schedules.md) · [34b Context parallelism, EP & 5D](34b-context-and-expert-parallelism.md) · [35 FP8 training & modern optimizers](35-fp8-training-and-optimizers.md)
- **Part VII — Post-Training & Alignment:** [36 SFT, chat templates & LoRA/QLoRA](36-sft-and-lora.md) · [36b Rejection sampling, Best-of-N & synthetic data](36b-rejection-sampling-and-synthetic-data.md) · [37 Reward modeling (BT, ORM, PRM) & PPO](37-reward-models-and-ppo.md) · [38 Direct alignment (DPO, IPO, SimPO, ORPO)](38-dpo-and-offline-preference.md) · [39 Critic-free RL (RLOO, GRPO, DAPO, GSPO, CISPO)](39-grpo-and-reinforcement-learning.md) · [40 Reasoning pipelines, thinking fusion & OPD](40-reasoning-pipelines-and-distillation.md) · [41 RL systems, TIS, AReaL & tool-use RL](41-rl-systems-inference-meets-training.md) · [42 Over-optimization, model merging & eval](42-overoptimization-merging-and-eval.md)

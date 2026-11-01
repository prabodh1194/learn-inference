# 41. RL infrastructure: where the inference engine meets the trainer

**Demo:** `book/code/rl_loop_profile.py` · **Build:** `train/rl_loop.py` · **Test:** `tests/test_41_rl_systems.py`
**Moves:** unifies Parts I–V and Parts VI–VII — explains why 75%–85% of RL training time is inference decode
**Prereq:** [01. The two phases](01-the-two-phases.md), [08. Continuous batching](08-continuous-batching.md), [10. Prefix caching](10-prefix-caching.md), [33. ZeRO and FSDP](33-zeRO-and-fsdp.md), [39. GRPO](39-grpo-and-reinforcement-learning.md)

---

## The problem

Look back at the two halves of this book:
- **Parts I–V (Inference):** memory-bandwidth-bound decode ($0.79\text{ ops:byte}$),
  KV caching ($112\text{ KiB/token}$), continuous batching, paged attention,
  prefix caching, speculative decoding, vLLM, and SGLang.
- **Parts VI–VII (Training & Post-Training):** compute-bound forward + backward
  passes ($6ND\text{–}8ND$ FLOPs), $16\Psi$ AdamW state, activation checkpointing,
  ZeRO/FSDP, and GRPO.

In an online RL loop (GRPO / PPO, Lectures 37 and 39), **every single training
step executes both halves back to back**:

```
   ┌──────────────────────────────────────────────────────────────┐
   │  PHASE 1: ROLLOUT (Inference Engine — vLLM / SGLang)         │
   │  For each prompt q, decode G = 8..64 completions token-by-   │
   │  token using current weights pi_theta_old (MEMORY-BOUND)     │
   └──────────────────────────────┬───────────────────────────────┘
                                  │  generated token ids + rewards
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │  PHASE 2: LEARNER UPDATE (Trainer — FSDP / Megatron)         │
   │  Run 1 forward + backward pass over all generated tokens in  │
   │  parallel and call optimizer.step() (COMPUTE-BOUND)          │
   └──────────────────────────────┬───────────────────────────────┘
                                  │  sync updated weights theta
                                  └───────────────────────────────► (repeat)
```

Ask Lecture 01's question of this loop: **which of the two phases dominates wall
clock time?**

---

## See it

```bash
uv run python book/code/rl_loop_profile.py
```

```
One GRPO step on RTX 3090 (Qwen3-0.6B, 4 prompts x G=8 = 32 sequences)
=========================================================================
  prompt_len   gen_len   rollout (decode)   learner (fwd+bwd)   rollout share
         256       512             1.51 s              2.44 s           38.2%
         256      2048            12.20 s              7.32 s           62.5%
         256      4096            40.85 s             13.82 s           74.7%
         256      8192           147.47 s             26.83 s           84.6%

  Rollout is Parts I-V (decode + KV cache). Learner is Part VI (6ND-8ND compute).
```

Read the `rollout share` column from top to bottom:
- At $512$-token responses, rollout is **38.2%** of the step.
- By $4{,}096$-token reasoning traces, rollout is **74.7%** of the step.
- By $8{,}192$-token reasoning traces (still only a quarter of DeepSeek-R1's
  $32\text{k}$–$65\text{k}$ rollout cap!), **84.6% of your RL training time is
  spent inside the inference engine generating tokens**, and only **15.4%** is
  spent computing gradients!

> **Modern RL post-training is 80% an inference problem.** Every optimization you
> built in Parts I–V sits directly on the critical path of training a reasoning
> model.

---

## The idea

### 1. Deriving the rollout-vs-learner time split from first principles

Let one GRPO step process $B_q = 4$ unique prompts, each with $G = 8$ group
completions, for a batch of $B = B_q \cdot G = 32$ sequences of prompt length
$P = 256$ and generated length $T_{\text{gen}} = 4096$ on an RTX 3090
($\text{BW} = 936.2\text{ GB/s}$, $\text{Peak} = 71\text{ TFLOPS}$, Qwen3-0.6B
active params $N = 440.4\text{M}$).

#### Phase 1: Rollout (autoregressive decode, $T_{\text{gen}} = 4096$ sequential steps)

Each of the $4{,}096$ decode steps must read from HBM:
1. **The model weights once per step** (shared across the $B = 32$ sequences,
   Lecture 07):
   $$2 N = 2 \times 440{,}401{,}920\text{ B} = 880.8\text{ MB}$$
2. **The KV cache of all $B = 32$ sequences at the current context length**
   (growing from $P = 256$ to $P + T_{\text{gen}} = 4352$, average context
   $\bar{C} = 256 + \frac{4096}{2} = 2{,}304$ tokens, at $114{,}688\text{ B/token}$
   from Lecture 05):
   $$\text{KV}_{\text{step}} = B \times \bar{C} \times 114{,}688\text{ B} = 32 \times 2{,}304 \times 114{,}688\text{ B} = 8{,}455.7\text{ MB}$$

Sum the two HBM reads per decode step and multiply by $T_{\text{gen}} = 4{,}096$
steps:

```
bytes per decode step =   880.8 MB (weights) + 8,455.7 MB (KV cache)  =  9,336.5 MB
over 4,096 steps      =   4,096 × 9,336.5 MB                          =  38.24 TB
rollout time @ 936.2 GB/s:  38.24 × 10^12 B / (936.2 × 10^9 B/s)      =  40.85 s
```

#### Phase 2: Learner update (one parallel pass over all $D$ tokens)

Once all $T_{\text{gen}} = 4{,}096$ tokens exist, the Learner processes all
$B = 32$ sequences of length $P + T_{\text{gen}} = 4{,}352$ **in parallel** (just
like prefill in Lecture 01!):

$$D = B \times (P + T_{\text{gen}}) = 32 \times 4{,}352 = 139{,}264 \text{ tokens}$$

With full activation checkpointing ($8ND$ FLOPs, Lecture 31) at $50\%$ HFU
($35.5\text{ TFLOP/s}$ effective on the 3090):

```
learner FLOPs =  8 × 440,401,920 × 139,264          =  490.66 TFLOP
learner time  =  490.66 TFLOP / (0.50 × 71 TFLOP/s) =   13.82 s
```

Compare the two phases:

$$\text{Rollout share} = \frac{40.85\text{ s}}{40.85\text{ s} + 13.82\text{ s}} = \frac{40.85}{54.67} = 74.7\%$$

---

### 2. Every Part II optimization is an RL training optimization

Look at what happens inside that 40.85-second rollout phase:

1. **Prefix caching (Lecture 10) is a $G\times$ win on every prompt.**
   In GRPO, $G = 8$ to $64$ completions are sampled for the **exact same prompt
   $q$**. With prefix caching enabled in vLLM / SGLang, $q$ is prefilled **once**
   and its physical KV blocks are shared across all $G$ branches (`ref_count = G`)!
2. **Continuous batching (Lecture 08) and the long-tail straggler problem.**
   In a group of $32 \times 16 = 512$ reasoning traces, 500 traces may finish in
   $2{,}000$ tokens while 12 stragglers keep thinking to $32{,}768$ tokens. By
    step $10{,}000$, your GPU is running at batch size $12$ instead of $512$!
   Production RL systems mitigate this with **continuous prompt-pool refilling**
   (or partial rollout continuation across steps).
3. **PagedAttention (Lecture 09) and KV quantization (Lecture 19).**
   At average context $2{,}304$ tokens across $32$ sequences, the KV cache
   ($8.46\text{ GB}$) is already **$9.6\times$ larger than the weights
   ($880.8\text{ MB}$)**! Without PagedAttention, the rollout phase OOMs at a
   fraction of the batch size.
4. **Speculative decoding / MTP (Lectures 12 & 35).**
   Because the Learner cannot start until the longest rollout finishes,
   repurposing DeepSeek-V3's MTP head for speculative decoding cuts the tail
   rollout latency nearly in half.

---

### 3. The RL systems dilemma: Co-located (HybridEngine) vs Disaggregated

How should you arrange the **Rollout Engine (vLLM/SGLang)** and the **Learner
(FSDP/Megatron)** across your GPUs?

They want **opposite parallelism layouts** and compete for the **same HBM**:

| Requirement | Rollout Engine (vLLM / SGLang) | Learner (FSDP / Megatron) |
|---|---|---|
| **Weights** | Unsharded BF16/FP8 per replica (or small TP) | Sharded across DP ranks ($16\Psi / D$, L33) |
| **Rest of VRAM** | **Paged KV cache pool** (as large as possible!) | **Optimizer states ($12\Psi$), grads ($2\Psi$), activations** |
| **Bottleneck** | Memory bandwidth | Tensor-core FLOPS |

Two architectures dominate open-source RL frameworks (**veRL**, **OpenRLHF**, **SkyRL**):

1. **Co-located (veRL `3D-HybridEngine`):** Both vLLM and FSDP live on the
   **same GPUs**, taking turns in time:
   - Before Rollout: FSDP runs an `all-gather` to reshard weights from the
     training mesh into vLLM's inference TP layout, then **offloads/sleeps**
     FSDP's gradients and optimizer states (or wakes vLLM's CUDA memory pool via
     `sleep_level=1`) so vLLM can claim $80\%$ of VRAM for its Paged KV cache.
   - Before Learner update: vLLM **sleeps its KV cache pool** (freeing the
     physical blocks without destroying the process or CUDA graphs), and FSDP
     wakes up to run the backward pass and `optimizer.step()`.
2. **Disaggregated Rollout / Learner (Ray actors on separate GPU pools):**
   One pool of GPUs runs vLLM continuously; another pool runs FSDP; updated
   weights are broadcast from rank 0 of FSDP to the vLLM workers over a
   dedicated NCCL process group (`init_process_group` between trainer and vLLM
   workers — see [Troubleshooting Incident I9](troubleshooting.md#i9-an-rl-trainers-vllm-integration-segfaults)
   for what happens when their PyTorch/vLLM versions drift!).

??? question "Why can't you just keep vLLM's KV cache pool allocated while FSDP runs the backward pass on the same GPU?"
    Because vLLM pre-allocates `gpu_memory_utilization` (default $90\%$) of VRAM
    for its Paged KV cache pool at startup (Lecture 09), while FSDP needs VRAM
    for the $16\Psi / D$ model states plus activation buffers and temporary
    `all-gather` buckets (Lectures 31 and 33). If both hold their pools at once,
    they exceed $100\%$ of VRAM and crash immediately. That is why modern RL
    engines use CUDA virtual memory management (`cuMemCreate` / vLLM sleep mode)
    to unmap vLLM's KV pages during the Learner step and unmap optimizer states
    during the Rollout step!
    [Full answer](qa.md#why-must-co-located-rl-trainers-sleep-vllms-kv-cache-during-the-backward-pass)

---

### 4. Learner–Sampler Numerical Mismatch, Truncated Importance Sampling (TIS) & Asynchronous RL (`AReaL`)

Even when the vLLM Sampler and the FSDP Learner hold the **exact same weight
tensor $\theta_{\text{old}}$**, their token log-probabilities do not match:

$$\pi_{\theta_{\text{old}}}^{\text{sampler}}(a_t \mid s_t) \neq \pi_{\theta_{\text{old}}}^{\text{learner}}(a_t \mid s_t)$$

Why? Because vLLM uses PagedAttention, FP8/BF16 decode kernels, and inference TP
reduction order, whereas FSDP uses packed varlen FlashAttention and training
TP/SP reduction order — and floating-point addition is non-associative
([Lecture 02](02-arithmetic-intensity.md)). Over an $8{,}192$-token rollout, small per-token
log-prob drifts compound into large sequence probability discrepancies (*"Your
framework secretly brings you off-policy RL"*, Yao et al. 2025).

**Truncated Importance Sampling (TIS, Ionides 2008; adopted in veRL, OpenRLHF, Open-Instruct)**
fixes this by recomputing $\pi_{\theta_{\text{old}}}^{\text{learner}}$ on the
learner so the PPO ratio $\rho_t^{\text{policy}} = \pi_\theta^{\text{learner}} / \pi_{\theta_{\text{old}}}^{\text{learner}}$
starts at $1.0000$ at gradient step 0, and multiplying the loss by a detached,
one-sided capped sampler-to-learner importance weight ($C = 2.0$):

$$\tilde{\rho}_t^{\text{learner}} = \min\!\left(\frac{\pi_{\theta_{\text{old}}}^{\text{learner}}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}^{\text{sampler}}(a_t \mid s_t)},\; C\right)$$

**Fully Asynchronous RL (`AReaL`, Fu et al. 2025).** In a synchronous RL step,
every GPU waits for the single longest $32\text{k}$-token rollout in the batch to
finish before starting the learner pass. **AReaL** decouples streaming rollout
workers from learner workers completely: rollout workers generate continuously
without barriers (using **interruptible rollouts** when new weights arrive,
resuming from the cached KV prefix), while learner workers update whenever a
batch buffer fills and apply a staleness-bounded PPO objective to achieve a
**$2.57\times$–$2.77\times$ throughput speedup** on 512 GPUs.

---

### 5. Multi-Turn Tool-Use and Agentic RL

Modern reasoning agents (`OpenAI o3`, `Qwen3`, `Kimi K2`) interleave `<think>`
tokens with external tool calls (`<code>...</code>` or JSON MCP `<tool_call>`)
and environment observations (`<tool_output>...</tool_output>`):

```
[Prompt x] -> [Asst: <think>...<code>fib(50)</code>] -> [Env: <output>12586269025</output>] -> [Asst: </think><answer>...]
   mask=0                     mask=1                                  mask=0                            mask=1
```

Two systems rules govern multi-turn tool-use RL:
1. **Mask environment/tool output tokens (`completion_mask = 0`):** The tokens
   inside `<output>...</output>` were produced by the Python interpreter or search
   API, **not** sampled from $\pi_\theta$. Leaving `completion_mask = 1` on tool
   outputs corrupts the policy gradient by training the model to hallucinate
   interpreter stdout and breaks importance sampling ratios $\rho_t$.
2. **Preserve KV cache across tool turns:** When the rollout worker pauses at
   `</code>`, executes the sandbox, and appends `<output>...</output>`, prefix
   caching (or paused request slots in SGLang/vLLM) reuses the KV blocks of all
   prior turns so a 10-turn SWE-bench trajectory does not re-prefill $O(K^2)$ tokens.

---

## The code

```python
def rl_step_time_breakdown(
    prompt_len: int,
    gen_len: int,
    group_size: int,
    n_prompts: int,
    active_params: int = 440_400_000,
    bandwidth: float = 936.2e9,
    peak_flops: float = 71e12,
) -> dict[str, float]:
    total_seqs = n_prompts * group_size
    total_tokens = total_seqs * (prompt_len + gen_len)
    rollout_s = (gen_len * 2.0 * active_params) / bandwidth
    learner_s = (6.0 * active_params * total_tokens) / peak_flops
    total_s = rollout_s + learner_s
    return {
        "rollout_s": rollout_s,
        "train_s": learner_s,
        "rollout_share": rollout_s / total_s,
    }
```

---

## Build it

1. Implement `rl_step_time_breakdown` in `train/rl_loop.py`.
2. Run the tests:

```bash
uv run pytest tests/test_41_rl_systems.py -v
```

3. Compare `rollout_share` at `gen_len = 128` vs `gen_len = 8192`, and explain why
   adding the KV-cache traffic term (as in `book/code/rl_loop_profile.py`) pushes
   `rollout_share` even higher at long contexts.

---

## What you should see

- **Rollout time grows with generated trace length** (and quadratically once KV
  cache traffic exceeds weight traffic past the ~8k crossover from Lecture 01!).
- **Every inference win in Parts I–V directly accelerates RL training in Part VII.**

---

## Go deeper

- **[HybridFlow: A Flexible and Efficient RLHF Framework (veRL)](https://arxiv.org/abs/2409.19256)** (Sheng et al., EuroSys 2025) — the `3D-HybridEngine` weight resharding and memory-pool sleep/wake design.
- **[OpenRLHF: An Easy-to-Scale RLHF Framework](https://arxiv.org/abs/2405.11143)** (Hu et al., 2024) — Ray + vLLM + DeepSpeed ZeRO-3 architecture.
- **[DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948)** — how rollout length scaling drove reasoning emergence.

---

## Check yourself

1. Why is the Rollout phase of GRPO memory-bandwidth-bound while the Learner
   update phase on the exact same tokens is compute-bound?
2. How does Lecture 10's prefix caching save work specifically on GRPO's $G$
   group completions per prompt?
3. Why do co-located RL frameworks (`veRL`) need to reshard weights and sleep
   vLLM's KV cache pool between the Rollout phase and the Learner phase?

---

## Next

**[42. Over-optimization, model merging, and post-training evaluation](42-overoptimization-merging-and-eval.md)**:
derive Goodhart's KL over-optimization peak, Assistant Axis activation capping, TIES/DARE model merging, and unbiased `pass@k` vs. `pass^k` evaluation.

```bash
uv run python book/code/merging_and_eval.py
```

# Appendix B: Papers

Indexed to the lecture that makes each one readable. **Read them in that order.**

A paper you've implemented reads completely differently from one you haven't,
you skip the motivation, argue with the design choices, and go straight to the
details you got wrong. That's the whole reason this book puts building first.

---

## Foundations

**[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** · Vaswani et al., 2017 · **L03**
The architecture. §3.2.3's causal masking is what makes KV caching valid.

**[Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785)** · Williams et al., 2009 · **L02**
Predates GPUs in this role and still the clearest statement of compute- vs.
memory-bound.

**[GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)** · Ainslie et al., 2023 · **L05**
Why Qwen3 has 16 query heads and 8 KV heads, a 2× cut in the thing that
bottlenecks decode.

---

## Serving architecture

**[Orca: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/conference/osdi22/presentation/yu)** · Yu et al., OSDI '22 · **L07–L08**
Continuous batching, which they call iteration-level scheduling. §2–3 motivates
the problem; the rest is the design you built.

**[Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)** · Kwon et al., SOSP '23 · **L09–L10**
The vLLM paper. §4 is the memory manager; §4.3 is the sharing that becomes prefix
caching. The OS analogy is drawn explicitly.

**[SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/abs/2308.16369)** · Agrawal et al., 2023 · **L11**
"Piggybacking" is the mechanism: prefill rides in capacity decode wasn't using.

**[Taming Throughput-Latency Tradeoff (Sarathi-Serve)](https://arxiv.org/abs/2403.02310)** · Agrawal et al., 2024 · **L11**
The follow-up, with stall-free scheduling.

**[SGLang: Efficient Execution of Structured LM Programs](https://arxiv.org/abs/2312.07104)** · Zheng et al., 2023 · **L10, L27**
RadixAttention, a radix tree rather than a hash map for prefix sharing. Better
for branching conversations.

---

## Speculative decoding

**[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)** · Leviathan et al., 2022 · **L12**
The original. §2.3's rejection-sampling proof is what makes it exact rather than
approximate, required before you claim distributional equivalence under sampling.

**[Medusa](https://arxiv.org/abs/2401.10774)** · Cai et al., 2024 · **L12**
Extra decoding heads instead of a draft model.

**[EAGLE](https://arxiv.org/abs/2401.15077)** · Li et al., 2024 · **L12, L29**
Feature-level drafting. The current practical default, and a good paper to
reproduce.

---

## Kernels

**[FlashAttention](https://arxiv.org/abs/2205.14135)** · Dao et al., 2022 · **L17**
§3.1 and Algorithm 1 are what you implement. Read after building it.

**[FlashAttention-2](https://arxiv.org/abs/2307.08691)** · Dao, 2023 · **L17, L29**
Better work partitioning. Explains where your Triton version loses.

**[Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)** · Milakov & Gimelshein, 2018 · **L17**
The running-max trick in isolation. Short, and the cleanest statement of it.

**[FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)** · 2023 · **L18**
Splitting context across thread blocks so batch-1 decode doesn't idle the GPU.

**[Triton: An Intermediate Language and Compiler](https://dl.acm.org/doi/10.1145/3315508.3329973)** · Tillet et al., 2019 · **L16**
Why block-level programming is the right abstraction.

**[Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)** · Harris · **L20**
Seven stages, each a memory-access insight. Still the best single document on why
access patterns dominate.

---

## Quantization

**[GPTQ](https://arxiv.org/abs/2210.17323)** · Frantar et al., 2022 · **L19**
Layer-wise second-order quantization.

**[AWQ](https://arxiv.org/abs/2306.00978)** · Lin et al., 2023 · **L19**
Not all weights matter equally, protect the salient ~1%. §3 is the argument.

**[SmoothQuant](https://arxiv.org/abs/2211.10438)** · Xiao et al., 2022 · **L19**
Shifting outlier difficulty from activations into weights.

---

## Parallelism

**[Megatron-LM](https://arxiv.org/abs/1909.08053)** · Shoeybi et al., 2019 · **L22**
§3 has the column/row split you implement. The original TP paper.

**[Switch Transformers](https://arxiv.org/abs/2101.03961)** · Fedus et al., 2021 · **L23**
The clearest MoE introduction: routing, capacity factors, load balancing.

**[Mixtral of Experts](https://arxiv.org/abs/2401.04088)** · Jiang et al., 2024 · **L23**
A real open MoE with inference details.

**[DeepSeek-V3](https://arxiv.org/abs/2412.19437)** · DeepSeek-AI, 2024 · **L23**
Fine-grained and shared experts; current state of the art in MoE inference design.

---

## Production

**[DistServe](https://arxiv.org/abs/2401.09670)** · Zhong et al., 2024 · **L27**
Prefill/decode disaggregation. §3 quantifies the interference it removes.

**[Splitwise](https://arxiv.org/abs/2311.18677)** · Patel et al., 2023 · **L27**
Same idea, arguing for different hardware per phase.

**[Open Versus Closed: A Cautionary Tale](https://www.usenix.org/legacy/event/nsdi06/tech/schroeder.html)** · Schroeder et al., NSDI '06 · **L25**
Not LLM-specific, directly applicable: closed-loop load tests cannot show
overload.

---

## Training systems

**[Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)** · Hoffmann et al., 2022 · **L30**
Why $C \approx 6ND$ and why $D \approx 20N$ is optimal for training cost alone
(before inference cost shifts the balance toward over-training small models).

**[Mixed Precision Training](https://arxiv.org/abs/1710.03740)** · Micikevicius et al., 2018 · **L30**
Why FP32 master weights are required when forward/backward run in 16-bit floats.

**[Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198)** · Korthikanti et al., 2022 · **L31, L34**
The exact per-layer activation memory formula ($34sbh + 5as^2b$), selective
activation checkpointing, and Sequence Parallelism ($g$ and $\bar{g}$).

**[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)** · Rajbhandari et al., 2019 · **L33**
ZeRO-1, ZeRO-2, and ZeRO-3 memory and communication derivations.

**[Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)** · Qi et al., 2024 · **L34**
Splitting backward into $B_x$ and $B_w$ so $B_w$ fills pipeline warmup/drain bubbles.

**[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)** · DeepSeek-AI, 2024 · **L23, L34, L35**
$1\times 128$ tile / $128\times 128$ block FP8 training, DualPipe, Multi-Token
Prediction, and auxiliary-loss-free MoE load balancing.

---

## Post-training & alignment

**[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** · Hu et al., 2021 · **L36**
Low-rank adapters, $B=0$ initialization, and $\alpha/r$ scaling.

**[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)** · Dettmers et al., 2023 · **L36**
4-bit NormalFloat (NF4) and double quantization.

**[Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)** · Ouyang et al., 2022 · **L37**
The classic SFT $\to$ Bradley-Terry Reward Model $\to$ PPO pipeline.

**[Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290)** · Rafailov et al., 2023 · **L38**
Closed-form extraction of the optimal policy that cancels the partition function $Z(x)$.

**[DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/abs/2402.03300)** · Shao et al., 2024 · **L39**
Introduces GRPO (deleting PPO's Critic via group-relative baselines) and the
unified gradient paradigm across SFT, RFT, DPO, PPO, and GRPO.

**[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948)** · DeepSeek-AI, 2025 · **L28b, L39, L40**
R1-Zero emergent reasoning via verifiable rewards, the 4-stage R1 pipeline, and
why distilling R1 into a 32B model beats direct RL on the 32B model by 25.6 AIME points.

**[HybridFlow: A Flexible and Efficient RLHF Framework (veRL)](https://arxiv.org/abs/2409.19256)** · Sheng et al., EuroSys 2025 · **L41**
Co-locating vLLM rollout engines and FSDP trainers via 3D-HybridEngine weight
resharding and KV pool sleep/wake.

---

## Books and long-form

**Philip Kiely, *Inference Engineering*** (Baseten, 2026)
The breadth-first survey this book indexes throughout. Appendix A is a good
glossary; Appendix B has its own curated reading list. **Chapter 6** covers
modalities this book skips entirely, VLM, embedding, ASR, TTS, image, video.

**Aleksa Gordić, [*Inside vLLM*](https://www.aleksagordic.com/blog/vllm)** · **L14**
Top-down read of vLLM V1. Save it for Lecture 14.

**Brendan Gregg, [*Systems Performance*](https://www.brendangregg.com/systems-performance-2nd-edition-book.html)**
Not about GPUs. The methodology chapters are the best available treatment of how
to think about performance work.

---

## Source worth reading

| Repo | Why |
|---|---|
| **[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)** | ~1,200 lines. The nanoGPT of inference. **L14** |
| **[vLLM](https://github.com/vllm-project/vllm)** | Production. Read `vllm/v1/core/` targeted, not whole. |
| **[SGLang](https://github.com/sgl-project/sglang)** | Different prefix-caching design. |
| **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** | Hand-written CUDA and Triton, side by side. |

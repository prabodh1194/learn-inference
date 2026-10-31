# 40. Reasoning pipelines (R1-Zero to R1) and knowledge distillation

**Demo:** `book/code/distillation_loss.py` · **Build:** `train/distill.py` · **Test:** `tests/test_40_distillation.py`
**Moves:** AIME pass@1 from 47.0% (direct RL on 32B) to 72.6% (distillation from R1); preserves gradient scale across temperature $T$
**Prereq:** [39. GRPO and verifiable rewards](39-grpo-and-reinforcement-learning.md), [28b. Reasoning and test-time compute](28b-reasoning-and-test-time-compute.md)

---

## The problem

In Lecture 39, we built GRPO with verifiable rewards (RLVR). When DeepSeek
pointed that loop directly at `DeepSeek-V3-Base` with **zero supervised
fine-tuning** (`DeepSeek-R1-Zero`), two things happened at once:

1. **Reasoning emerged on its own.** AIME 2024 pass@1 climbed from **15.6% to
   77.9%** (and **86.7%** with 64-sample majority voting). Average response
   length grew from hundreds to thousands of tokens as the model discovered
   self-verification, backtracking, and reflection (*"Wait, wait. That's an aha
   moment..."*) purely because thinking longer got the final answer right more
   often.
2. **The model became a terrible product.** Because the reward only checked
   whether the boxed final answer was right and inside `<think>` tags, everything
   the reward didn't measure rotted: its chains of thought were sprawling and
   hard to read, it switched between English and Chinese mid-sentence
   (**language mixing**), and it couldn't handle general writing or chat.

And a second puzzle appeared when they tried to produce small models (1.5B to
32B) that people can actually run on one GPU: **running the exact same large-scale
RL loop directly on `Qwen-32B-Base` stalled at 47.0% on AIME — whereas simple SFT
distillation on R1's traces reached 72.6%!**

---

## See it

```bash
uv run python book/code/distillation_loss.py
```

```
Temperature-scaled KL distillation: why the T^2 factor is required
=========================================================================
  temp (T)    unscaled KL    T^2 * KL    unscaled grad ||dKL/dz_s||   T^2-scaled grad
       1.0        0.17783      0.1778                       0.36446             0.36446
       2.0        0.06321      0.2528                       0.10882             0.43527
       5.0        0.00960      0.2399                       0.01552             0.38804
      10.0        0.00219      0.2188                       0.00352             0.35197

DeepSeek-R1 Section 4.2 ablation on Qwen-32B-Base (discovery vs imitation)
=========================================================================
  method                                      AIME 2024 pass@1   MATH-500 pass@1
  Qwen-32B-Base + 10k steps of direct RL          47.0%              83.0%
  Qwen-32B-Base + SFT distillation from R1        72.6%              94.3%
```

Look at both tables:
- Without the $T^2$ multiplier in logit distillation, raising temperature from
  $T = 1.0$ to $T = 10.0$ causes the gradient norm $\|\nabla_{z_S} \text{KL}\|$
  to collapse by **$103\times$** ($0.36446 \to 0.00352$, scaling as $1/T^2$!).
  Multiplying by $T^2$ keeps the gradient norm invariant ($\approx 0.35$–$0.43$).
- On `Qwen-32B-Base`, copying a frontier teacher's reasoning traces beats
  rediscovering reasoning via direct RL by **+25.6 points on AIME** ($47.0\% \to 72.6\%$).

---

## The idea

### 1. The 4-stage DeepSeek-R1 pipeline: each stage repairs what the previous stage broke

Why did `DeepSeek-R1` need **four stages** instead of `R1-Zero`'s single RL stage?
Trace the failure mode that each stage exists to fix:

```
  DeepSeek-V3-Base
         │
         ▼  Stage 1: Cold-Start SFT (~thousands of readable long-CoT traces)
         │           Fixes: feral formatting & readability before RL explores
         ▼  Stage 2: Reasoning GRPO + Language-Consistency Reward
         │           Fixes: English/Chinese mixing inside CoT; grows reasoning
         ▼  Stage 3: Rejection Sampling SFT (~600k reasoning + ~200k general)
         │           Fixes: converts hard-to-steer RL policy into clean SFT data + general skills
         ▼  Stage 4: All-Scenario RL (rule rewards for math/code + RM for helpful/safe)
         │           Fixes: general helpfulness & safety alignment without reward-hacking CoT
         ▼
    DeepSeek-R1
```

Walk through the engineering logic of each transition:

1. **Stage 1 (Cold-Start SFT):** Instead of starting RL from a raw base model
   that babbles, fine-tune on a small seed (a few thousand examples) of
   cleanly formatted `<think>...\n</think>\nsummary` traces. That plants a
   readable voice so RL explores inside a human-legible format.
2. **Stage 2 (Reasoning GRPO + Language Consistency):** Run GRPO (Lecture 39) on
   math, code, and STEM problems, adding a tiny rule-based reward:
   $$R_{\text{lang}} = \frac{\text{count}(\text{words in target language})}{\text{count}(\text{all words})}$$
   Wait: the authors note in their ablation that $R_{\text{lang}}$ costs a tiny
   dip in raw math score — and they pay it deliberately because an unreadable
   polyglot chain-of-thought is unusable.
3. **Stage 3 (Rejection Sampling Fine-Tuning, RFT):** The Stage 2 RL checkpoint
   is a world-class reasoner, but it only knows reasoning prompts. Instead of
   shipping it directly, use it as a **synthetic data factory**: sample many
   candidate solutions per prompt, **filter** them with verifiers and a
   generative judge to keep only the correct, clean ones ($\sim 600\text{k}$
   reasoning traces), mix in $\sim 200\text{k}$ general non-reasoning SFT
   examples (writing, QA, self-cognition), and train a fresh `DeepSeek-V3-Base`
   for 2 epochs on the combined $800\text{k}$ dataset.
4. **Stage 4 (All-Scenario RL):** Run a final RL pass (1,700 steps, applying
   neural preference reward models only in the final 400 steps so the reward
   model cannot get hacked!).

---

### 2. Why distillation beats direct RL on small models: exploration vs imitation

Why did 10,000 steps of direct GRPO on `Qwen-32B-Base` stall at **47.0% AIME**,
while simple SFT on R1's $800\text{k}$ rejection-sampled traces jumped to **72.6%**?

Look at the search problem each algorithm faces:

- **Direct RL (exploration):** To reinforce a good reasoning strategy under a
  binary $0/1$ outcome verifier, the model must first **sample a trace that
  lands the right answer by chance** within its group of $G$ rollouts. If a hard
  AIME problem requires a 4,000-token chain of algebraic substitutions and a 32B
  base model has near-zero prior probability of stringing all 4,000 tokens
  together correctly, all $G$ samples score $r_i = 0 \implies \hat{A}_i = 0$
  (Lecture 39!). The gradient is zero; the model learns nothing.
- **Distillation (imitation):** The 671B teacher has *already paid* the massive
  exploration bill to find the needle in the $151{,}936^{4000}$ haystack. Once
  that 4,000-token trace is written down, SFT/distillation gives the 32B student
  **dense, token-by-token supervision at every single one of the 4,000 steps** —
  even on problems the student would never have solved on its own!

> **Discovery is expensive; copying discovery is cheap.** At the frontier (671B),
> RL is how new reasoning behaviors are discovered; below the frontier (1.5B–32B),
> distillation is how they are transferred.

---

### 3. Sequence-level SFT vs Logit-level KL distillation (and the $T^2$ derivation)

Distillation comes in two forms:

1. **Sequence-level (black-box) distillation:** sample complete text traces
   $y \sim \pi_{\text{teacher}}(\cdot \mid x)$ and run standard SFT
   cross-entropy on the student $\pi_\theta$. Requires only text outputs (works
   even across different tokenizers!).
2. **Logit-level (white-box) knowledge distillation (Hinton et al., 2015):** at
   each token position, match the student's full vocabulary distribution
   $p_S^{(T)} = \text{softmax}(z_S / T)$ to the teacher's softened distribution
   $p_T^{(T)} = \text{softmax}(z_T / T)$ by minimizing the KL divergence scaled
   by $T^2$:

$$\mathcal{L}_{\text{KD}}(z_S, z_T; T) = T^2 \, \mathbb{D}_{\text{KL}}\bigl(p_T^{(T)} \,\big\|\, p_S^{(T)}\bigr) = T^2 \sum_{v=1}^V p_{T,v}^{(T)} \log \frac{p_{T,v}^{(T)}}{p_{S,v}^{(T)}}$$

#### Why the $T^2$ factor is required — derived step by step

Let $z_S, z_T \in \mathbb{R}^V$ be the student and teacher logit vectors at one
token position, and define the temperature-softened probabilities:

$$p_{S,i}^{(T)} = \frac{e^{z_{S,i} / T}}{\sum_k e^{z_{S,k} / T}}, \qquad p_{T,i}^{(T)} = \frac{e^{z_{T,i} / T}}{\sum_k e^{z_{T,k} / T}}$$

**Step 1: Differentiate the unscaled KL divergence $\mathbb{D}_{\text{KL}}(p_T^{(T)} \,\|\, p_S^{(T)})$ with respect to a student logit $z_{S,j}$.**
Since $\sum_i p_{T,i}^{(T)} \log p_{T,i}^{(T)}$ does not depend on $z_S$, we only
differentiate $-\sum_i p_{T,i}^{(T)} \log p_{S,i}^{(T)}$:

$$\frac{\partial}{\partial z_{S,j}}\Bigl[\mathbb{D}_{\text{KL}}\bigl(p_T^{(T)} \,\big\|\, p_S^{(T)}\bigr)\Bigr] = -\frac{\partial}{\partial z_{S,j}} \sum_{i=1}^V p_{T,i}^{(T)} \left(\frac{z_{S,i}}{T} - \log \sum_{k=1}^V e^{z_{S,k}/T}\right)$$

Carry out the derivative term by term (using $\sum_i p_{T,i}^{(T)} = 1$):

$$= -\frac{1}{T} p_{T,j}^{(T)} + \left(\sum_{i=1}^V p_{T,i}^{(T)}\right) \frac{\frac{1}{T} e^{z_{S,j}/T}}{\sum_{k=1}^V e^{z_{S,k}/T}} = \frac{1}{T}\bigl(p_{S,j}^{(T)} - p_{T,j}^{(T)}\bigr)$$

**Step 2: Expand $p_{S,j}^{(T)} - p_{T,j}^{(T)}$ in the high-temperature regime ($T \gg |z_i|$).**
Using the first-order Taylor expansion $e^{z_j / T} \approx 1 + \frac{z_j}{T}$
(and assuming zero-mean logits $\sum_k z_k = 0$ so $\sum_k e^{z_k / T} \approx V$):

$$p_{S,j}^{(T)} \approx \frac{1 + z_{S,j}/T}{V}, \qquad p_{T,j}^{(T)} \approx \frac{1 + z_{T,j}/T}{V}$$

Substitute both into our gradient from Step 1:

$$\frac{\partial}{\partial z_{S,j}}\Bigl[\mathbb{D}_{\text{KL}}\bigl(p_T^{(T)} \,\big\|\, p_S^{(T)}\bigr)\Bigr] = \frac{1}{T}\bigl(p_{S,j}^{(T)} - p_{T,j}^{(T)}\bigr) \approx \frac{1}{T} \cdot \frac{z_{S,j} - z_{T,j}}{V T} = \frac{z_{S,j} - z_{T,j}}{V \, T^2}$$

Look at the denominator: **$T^2$**! One factor of $1/T$ came from the chain
rule $\frac{\partial (z_S / T)}{\partial z_S} = \frac{1}{T}$, and the second
factor of $1/T$ came from the probabilities $p_S^{(T)}, p_T^{(T)}$ flattening
toward uniform $1/V$ at rate $1/T$. Without multiplying the loss by $T^2$,
raising $T$ from $1$ to $10$ shrinks the distillation gradient by $100\times$!

??? question "What is the difference between Forward KL $D_{\text{KL}}(p_T \,\|\, p_S)$ and Reverse KL $D_{\text{KL}}(p_S \,\|\, p_T)$ when distilling into a smaller model?"
    In **Forward KL** $\sum_y p_T(y) \log(p_T(y) / p_S(y))$, anywhere $p_T(y) > 0$,
    $p_S(y)$ is heavily penalized if it approaches $0$ (**mode-covering**): a
    small student that lacks capacity to fit all of $p_T$'s modes is forced to
    spread probability mass across the empty space *between* modes. In **Reverse
    KL** $\sum_y p_S(y) \log(p_S(y) / p_T(y))$ (used in MiniLLM and on-policy
    distillation), whenever $p_S(y) = 0$ the term is $0$, so the small student
    can **lock onto one high-quality mode** of the teacher without smearing
    (**mode-seeking**).
    [Full answer](qa.md#what-is-the-difference-between-forward-kl-and-reverse-kl-in-distillation)

---

## The code

```python
def kd_kl_loss(teacher_logits, student_logits, temperature: float = 1.0) -> float:
    import torch
    import torch.nn.functional as F

    z_t = teacher_logits.float() / temperature
    z_s = student_logits.float() / temperature
    p_t = F.softmax(z_t, dim=-1)
    log_p_t = F.log_softmax(z_t, dim=-1)
    log_p_s = F.log_softmax(z_s, dim=-1)
    kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean()
    return float((temperature**2) * kl.item())


def rejection_sample_filter(
    samples: list[tuple[str, float]], threshold: float = 1.0
) -> list[str]:
    return [text for text, score in samples if score >= threshold]
```

---

## Build it

1. Implement `kd_kl_loss` and `rejection_sample_filter` in `train/distill.py`.
2. Run the tests:

```bash
uv run pytest tests/test_40_distillation.py -v
```

3. Verify that `kd_kl_loss(z, z, temperature=5.0) == 0.0` and that scaling `temperature`
   keeps `kd_kl_loss` within the same order of magnitude on non-identical logits.

---

## What you should see

- **Multiplying by $T^2$ keeps the distillation loss and gradient O(1) across temperatures.**
- **Rejection sampling converts a verifier into a dataset filter**, keeping only
  completions above `threshold`.

---

## Go deeper

- **[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)** (DeepSeek-AI, 2025) — Sections 2, 3, and 4.2.
- **[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)** (Hinton et al., 2015) — the original $T^2$ logit-distillation derivation.
- **[MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543)** (Gu et al., 2023) — reverse KL vs forward KL for autoregressive LMs.

---

## Check yourself

1. Name the two failure modes of `DeepSeek-R1-Zero` that motivated the 4-stage
   `DeepSeek-R1` pipeline.
2. Why did direct RL on `Qwen-32B-Base` stall at 47.0% AIME while SFT distillation
   from R1 reached 72.6%?
3. Where do the **two** factors of $1/T$ come from in
   $\frac{\partial}{\partial z_S}\mathbb{D}_{\text{KL}}(p_T^{(T)} \,\|\, p_S^{(T)}) \propto \frac{1}{T^2}$?

---

## Next

**[41. RL infrastructure: where the inference engine meets the trainer](41-rl-systems-inference-meets-training.md)**:
the final lecture. Why 75%–85% of a modern RL training run is spent inside
vLLM/SGLang decoding rollouts — bringing Parts I–V and Parts VI–VII together into
one system.

```bash
uv run python book/code/rl_loop_profile.py
```

# 40. Reasoning pipelines (R1-Zero to R1) and knowledge distillation

**Demo:** `book/code/distillation_loss.py` · **Build:** `train/distill.py` · **Test:** `tests/test_40_distillation.py`
**Moves:** AIME pass@1 from 47.0% (direct RL on 32B) to 72.6% (distillation from R1); preserves gradient scale across temperature $T$
**Prereq:** [39. GRPO and verifiable rewards](39-grpo-and-reinforcement-learning.md), [28b. Reasoning and test-time compute](28b-reasoning-and-test-time-compute.md)

---

## The problem

In Lecture 39, we built GRPO with verifiable rewards (RLVR). When DeepSeek
pointed that loop directly at `DeepSeek-V3-Base` with **zero supervised
fine-tuning** (`DeepSeek-R1-Zero`), two things happened at once. First,
**reasoning emerged on its own**. AIME 2024 pass@1 climbed from **15.6% to
77.9%** (and **86.7%** with 64-sample majority voting). Average response length
grew from hundreds to thousands of tokens as the model discovered
self-verification, backtracking, and reflection (*"Wait, wait. That's an aha
moment..."*) simply because thinking longer got the final answer right more
often.

Second, **the model became a poor product**. Because the reward only checked
whether the boxed final answer was right and inside `<think>` tags, everything
the reward did not measure suffered: its chains of thought were sprawling and
hard to read, it switched between English and Chinese mid-sentence (**language
mixing**), and it struggled with general writing and chat.

Another puzzle appeared when the team tried to train small models (1.5B to 32B)
that fit on one GPU: **running the exact same large-scale RL loop directly on
`Qwen-32B-Base` stalled at 47.0% on AIME, whereas simple SFT distillation on
R1's traces reached 72.6%**.

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

Look at both tables. Without the $T^2$ multiplier in logit distillation, raising
temperature from $T = 1.0$ to $T = 10.0$ causes the gradient norm
$\|\nabla_{z_S} \text{KL}\|$ to shrink by **$103\times$** ($0.36446 \to 0.00352$,
scaling as $1/T^2$). Multiplying by $T^2$ keeps the gradient norm steady around
$0.35$ to $0.43$. And on `Qwen-32B-Base`, copying a frontier teacher's reasoning
traces beats rediscovering reasoning via direct RL by **+25.6 points on AIME**
($47.0\% \to 72.6\%$).

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

Walk through the engineering logic of each transition. In **Stage 1 (Cold-Start
SFT)**, instead of starting RL from a raw base model that babbles, we fine-tune
on a small seed of a few thousand cleanly formatted `<think>...\n</think>\nsummary`
traces. That plants a readable voice so RL explores inside a human-legible
format.

In **Stage 2 (Reasoning GRPO with a target-language reward)**, we run GRPO
(Lecture 39) on math, code, and STEM problems, adding a small rule-based reward
that measures the share of words written in the target language.

$$R_{\text{lang}} = \frac{\text{count}(\text{words in target language})}{\text{count}(\text{all words})}$$

The authors note in their ablation that $R_{\text{lang}}$ costs a slight dip in
raw math score, and they pay it on purpose because a polyglot chain-of-thought
is unreadable to users.

In **Stage 3 (Rejection Sampling Fine-Tuning, RFT)**, the Stage 2 RL checkpoint
is a strong reasoner, but it only knows reasoning prompts. Instead of shipping
it directly, we use it as a **synthetic data factory**: we sample many candidate
solutions per prompt, filter them with verifiers and a generative judge to keep
only the right, clean ones ($\sim 600\text{k}$ reasoning traces), mix in
$\sim 200\text{k}$ general non-reasoning SFT examples (writing, QA,
self-cognition), and train a fresh `DeepSeek-V3-Base` for 2 epochs on the
combined $800\text{k}$ dataset.

In **Stage 4 (All-Scenario RL)**, we run a final RL pass of 1,700 steps, turning
on the neural preference reward models only in the final 400 steps so the reward
model cannot get hacked.

Having separate chat (`GPT-4o`) and reasoning (`o1`) models doubles serving
deployment cost. **Qwen3** fuses both modes into a single checkpoint using a
similar 4-stage pipeline (Long-CoT Cold Start to Reasoning RL to Thinking Mode
Fusion SFT to General RL). During Stage 3 (**Thinking Mode Fusion**), the Stage
2 reasoning model is fine-tuned on a mix of rejection-sampled thinking traces
tagged `/think` and general chat traces tagged `/no_think`. Even `/no_think`
responses still emit an **empty thinking block** `<think>\n\n</think>` before
the final answer so the internal output format stays identical in both modes.
Because the model learns both full `<think>` blocks and empty `<think>` blocks,
it also supports a **user-controlled thinking budget**: if a `<think>` trace hits
$B_{\text{think}}$ tokens (say $2{,}048$), the serving engine appends a closing
`</think>\n\n` prompt and the model summarizes its partial reasoning into a
final answer.

---

### 2. Why distillation beats direct RL on small models: exploration vs imitation

Why did 10,000 steps of direct GRPO on `Qwen-32B-Base` stall at **47.0% AIME**,
while simple SFT on R1's $800\text{k}$ rejection-sampled traces jumped to **72.6%**?

Compare the search problem each method faces. Under **direct RL (exploration)**
with a binary $0/1$ outcome verifier, the model can only reinforce a good
reasoning strategy after it **samples a trace that lands the right answer by
chance** within its group of $G$ rollouts. If a hard AIME problem needs a
4,000-token chain of algebraic substitutions and a 32B base model has near-zero
prior probability of stringing all 4,000 tokens together right, all $G$ samples
score $r_i = 0 \implies \hat{A}_i = 0$ (Lecture 39). The gradient is zero, and
the model learns nothing.

Under **distillation (imitation)**, the 671B teacher has *already paid* the
exploration bill to find the needle in the $151{,}936^{4000}$ haystack. Once
that 4,000-token trace is written down, SFT or distillation gives the 32B
student **dense, token-by-token supervision at every single one of the 4,000
steps**, even on problems the student would never have solved on its own.

> **Discovery is expensive; copying discovery is cheap.** At the frontier (671B),
> RL is how new reasoning behaviors are discovered; below the frontier (1.5B to
> 32B), distillation is how they are transferred.

---

### 3. Sequence-level SFT vs Logit-level KL distillation (and the $T^2$ derivation)

Distillation comes in two forms. In **sequence-level (black-box) distillation**,
we sample complete text traces $y \sim \pi_{\text{teacher}}(\cdot \mid x)$ and
run standard SFT cross-entropy on the student $\pi_\theta$. That requires only
text outputs and even works when the teacher and student use different
tokenizers.

In **logit-level (white-box) knowledge distillation (Hinton et al., 2015)**, at
each token position we match the student's full vocabulary distribution
$p_S^{(T)} = \text{softmax}(z_S / T)$ to the teacher's softened distribution
$p_T^{(T)} = \text{softmax}(z_T / T)$ by minimizing the KL divergence scaled by
$T^2$.

$$\mathcal{L}_{\text{KD}}(z_S, z_T; T) = T^2 \, \mathbb{D}_{\text{KL}}\bigl(p_T^{(T)} \,\big\|\, p_S^{(T)}\bigr) = T^2 \sum_{v=1}^V p_{T,v}^{(T)} \log \frac{p_{T,v}^{(T)}}{p_{S,v}^{(T)}}$$

#### Why the $T^2$ factor is required, derived step by step

Let $z_S, z_T \in \mathbb{R}^V$ be the student and teacher logit vectors at one
token position, and define the temperature-softened probabilities:

$$p_{S,i}^{(T)} = \frac{e^{z_{S,i} / T}}{\sum_k e^{z_{S,k} / T}}, \qquad p_{T,i}^{(T)} = \frac{e^{z_{T,i} / T}}{\sum_k e^{z_{T,k} / T}}$$

In **Step 1**, differentiate the unscaled KL divergence
$\mathbb{D}_{\text{KL}}(p_T^{(T)} \,\|\, p_S^{(T)})$ with respect to a student
logit $z_{S,j}$. Because $\sum_i p_{T,i}^{(T)} \log p_{T,i}^{(T)}$ does not
depend on $z_S$, we only differentiate $-\sum_i p_{T,i}^{(T)} \log p_{S,i}^{(T)}$.

$$\frac{\partial}{\partial z_{S,j}}\Bigl[\mathbb{D}_{\text{KL}}\bigl(p_T^{(T)} \,\big\|\, p_S^{(T)}\bigr)\Bigr] = -\frac{\partial}{\partial z_{S,j}} \sum_{i=1}^V p_{T,i}^{(T)} \left(\frac{z_{S,i}}{T} - \log \sum_{k=1}^V e^{z_{S,k}/T}\right)$$

Carry out the derivative term by term, using $\sum_i p_{T,i}^{(T)} = 1$.

$$= -\frac{1}{T} p_{T,j}^{(T)} + \left(\sum_{i=1}^V p_{T,i}^{(T)}\right) \frac{\frac{1}{T} e^{z_{S,j}/T}}{\sum_{k=1}^V e^{z_{S,k}/T}} = \frac{1}{T}\bigl(p_{S,j}^{(T)} - p_{T,j}^{(T)}\bigr)$$

In **Step 2**, expand $p_{S,j}^{(T)} - p_{T,j}^{(T)}$ in the high-temperature
regime where $T \gg |z_i|$. Using the first-order Taylor expansion
$e^{z_j / T} \approx 1 + \frac{z_j}{T}$ and zero-mean logits $\sum_k z_k = 0$ so
$\sum_k e^{z_k / T} \approx V$, we get the linearized probabilities.

$$p_{S,j}^{(T)} \approx \frac{1 + z_{S,j}/T}{V}, \qquad p_{T,j}^{(T)} \approx \frac{1 + z_{T,j}/T}{V}$$

Substitute both into our gradient from Step 1.

$$\frac{\partial}{\partial z_{S,j}}\Bigl[\mathbb{D}_{\text{KL}}\bigl(p_T^{(T)} \,\big\|\, p_S^{(T)}\bigr)\Bigr] = \frac{1}{T}\bigl(p_{S,j}^{(T)} - p_{T,j}^{(T)}\bigr) \approx \frac{1}{T} \cdot \frac{z_{S,j} - z_{T,j}}{V T} = \frac{z_{S,j} - z_{T,j}}{V \, T^2}$$

Look at the denominator: **$T^2$**. One factor of $1/T$ came from the chain rule
$\frac{\partial (z_S / T)}{\partial z_S} = \frac{1}{T}$, and the second factor
of $1/T$ came from the probabilities $p_S^{(T)}, p_T^{(T)}$ flattening toward
uniform $1/V$ at rate $1/T$. Without multiplying the loss by $T^2$, raising $T$
from $1$ to $10$ shrinks the distillation gradient by $100\times$.

??? question "What is the difference between Forward KL $D_{\text{KL}}(p_T \,\|\, p_S)$ and Reverse KL $D_{\text{KL}}(p_S \,\|\, p_T)$ when distilling into a smaller model?"
    In **Forward KL** $\sum_y p_T(y) \log(p_T(y) / p_S(y))$, anywhere $p_T(y) > 0$,
    $p_S(y)$ is heavily penalized if it approaches $0$ (**mode-covering**): a
    small student that lacks capacity to fit all of $p_T$'s modes is forced to
    spread probability mass into the empty space *between* modes. In **Reverse
    KL** $\sum_y p_S(y) \log(p_S(y) / p_T(y))$ (used in MiniLLM and on-policy
    distillation), whenever $p_S(y) = 0$ the term is $0$, so the small student
    can **lock onto one high-quality mode** of the teacher without smearing
    (**mode-seeking**).
    [Full answer](qa.md#what-is-the-difference-between-forward-kl-and-reverse-kl-in-distillation)

---

### 4. Why SFT memorizes while RL generalizes, and On-Policy Distillation (OPD)

Controlled studies (Chu et al., 2025; Chen et al., 2025) show that sequential
SFT causes much worse catastrophic forgetting of prior skills than RL. Both
facts follow from the exact KL divergence each objective minimizes.

First, **SFT minimizes Forward KL $\mathbb{D}_{\text{KL}}(\pi_\star \,\|\, \pi_\theta)$**
against the empirical dataset distribution $\pi_\star$:

$$\mathbb{D}_{\text{KL}}(\pi_\star \,\|\, \pi_\theta) = \mathbb{E}_{(x, y) \sim \pi_\star}\bigl[\log \pi_\star(y \mid x) - \log \pi_\theta(y \mid x)\bigr] = \underbrace{-H(\pi_\star)}_{\text{const in }\theta} + \mathcal{L}_{\text{SFT}}(\theta)$$

Because samples $(x, y)$ come from a **fixed offline dataset $\pi_\star$**,
mode-covering pressure forces $\pi_\theta$ to stretch probability mass toward
the dataset's surface tokens, pulling mass away from the model's existing skills.

Second, **KL-regularized RL minimizes Reverse KL $\mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_\star)$**,
where $\pi_\star(y \mid x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y \mid x)\exp\bigl(\frac{1}{\beta}r(x, y)\bigr)$
is the Gibbs optimal policy from Lecture 38. Expand the Reverse KL under
**on-policy samples $y \sim \pi_\theta(\cdot \mid x)$**:

$$\begin{aligned}
\mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_\star)
&= \mathbb{E}_{y \sim \pi_\theta}\left[\log \pi_\theta(y \mid x) - \log \pi_{\text{ref}}(y \mid x) + \log Z(x) - \frac{1}{\beta}r(x, y)\right] \\
&= -\frac{1}{\beta}\Bigl(\mathbb{E}_{y \sim \pi_\theta}[r(x, y)] - \beta\,\mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})\Bigr) + \log Z(x) = -\frac{1}{\beta}\mathcal{J}_{\text{RL}}(\theta) + \text{const}
\end{aligned}$$

Because RL samples from the student's **own** policy $\pi_\theta$, updates stay
local to modes $\pi_\theta$ visits on its own.

Offline distillation also trains the student on teacher prefixes
$u_{<t} \sim \pi_T$, which introduces **exposure bias**: at test time, a
per-step imitation error $\epsilon$ compounds quadratically as $O(\epsilon L^2)$
over a length-$L$ rollout once the student drifts into states the teacher never
visited (Ross et al., 2011 DAgger). **On-Policy Distillation (OPD)**, used in
`Qwen3` lightweight models, `MiMo-V2-Flash`, and `DeepSeek-V4`, fixes that drift
by sampling trajectories **on-policy from the student**
$a \sim \pi_\theta(\cdot \mid s)$ and grading every visited token
$s_t = (s, a_{<t})$ with the teacher's dense log-probability gap as a
token-level RL advantage.

$$A_t^{\text{OPD}} = \log \pi_T(a_t \mid s_t) - \log \pi_\theta(a_t \mid s_t)$$

That cuts trajectory error compounding from $O(\epsilon L^2)$ to $O(\epsilon L)$
and takes only **$1/10$ of the GPU-hours** of a full 4-stage RL pipeline (Qwen3
§4.5). When $K$ domain-specialist teachers $\pi_{T_1}, \dots, \pi_{T_K}$ (math,
coding, tool-use, chat) are combined into one student via prompt-routed weights
$w_k(s)$, this becomes **Multi-Teacher On-Policy Distillation (MOPD)**:

$$\mathcal{L}_{\text{MOPD}}(\theta) = \mathbb{E}_{s,\, a \sim \pi_\theta(\cdot \mid s)} \sum_{t=1}^{|a|} \sum_{k=1}^K w_k(s)\,\mathbb{D}_{\text{KL}}\bigl(\pi_\theta(\cdot \mid s_t) \,\big\|\, \pi_{T_k}(\cdot \mid s_t)\bigr)$$

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

3. Verify that `kd_kl_loss(z, z, temperature=5.0) == 0.0` and that scaling
   `temperature` keeps `kd_kl_loss` within the same order of magnitude on
   non-identical logits.

---

## What you should see

Multiplying the KL loss by **$T^2$ keeps both the loss and the logit gradient
$O(1)$ across temperatures**. On sequence data, rejection sampling turns a
deterministic verifier into a dataset filter that keeps only completions scoring
at or above `threshold`.

---

## Go deeper

- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (DeepSeek-AI, 2025), Sections 2, 3, and 4.2.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015), the $T^2$ logit-distillation derivation.
- [MiniLLM: Knowledge Distillation of Large Language Models](https://arxiv.org/abs/2306.08543) (Gu et al., 2023), reverse KL vs forward KL for autoregressive LMs.

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
why 75% to 85% of a modern RL training run is spent inside vLLM and SGLang
decoding rollouts, bringing Parts I to V and Parts VI to VII together into one
system.

```bash
uv run python book/code/rl_loop_profile.py
```

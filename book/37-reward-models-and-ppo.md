# 37. Reward modeling and RLHF with PPO (the 4-model problem)

**Demo:** `book/code/rlhf_memory.py` · **Build:** `train/ppo_rlhf.py` · **Test:** `tests/test_37_ppo_rlhf.py`
**Moves:** aligns models to comparative preferences, and shows why PPO's $36\Psi$ memory wall led to DPO and GRPO
**Prereq:** [36. SFT, LoRA, and QLoRA](36-sft-and-lora.md), [06. Sampling](06-sampling.md)

---

## The problem

Supervised fine-tuning (Lecture 36) hits three ceilings. First, **it has no
negative signal**. Cross-entropy pushes up the probability of the target token
$y_t^*$; it only pushes down other tokens in proportion to what the model
already thought was likely. It cannot tell the model that a specific plausible
hallucination or sycophantic answer is bad.

Second, **writing demonstrations is harder than judging them**. Asking a human
to write an optimal 500-line Python patch from scratch is slow and caps the
model at the annotator's skill. Asking a human or a test suite which of two
model outputs $y_w$ or $y_l$ is better is fast and scales.

Third, **SFT suffers from exposure bias**. SFT trains only on ground-truth
prefixes (teacher forcing), whereas at inference time the model conditions on
its *own* sampled tokens.

The classic fix, **Reinforcement Learning from Human Feedback (RLHF)** with
**Proximal Policy Optimization (PPO)** from Ouyang et al. (*InstructGPT*, 2022),
solves those three problems and creates a systems crisis of its own. It must
hold **four full language models in memory at the same time ($36\Psi$ bytes)**
and train a per-token **Critic (Value network)** whose job is nearly impossible
when rewards only arrive at the end of the response.

---

## See it

```bash
uv run python book/code/rlhf_memory.py
```

```
Model-state VRAM bill for Qwen3-0.6B (0.60B params)
=========================================================================
  SFT / Pre-training         Actor(16P)                                     8.88 GiB
  PPO RLHF (4 models)        Actor(16P) + Critic(16P) + Ref(2P) + RM(2P)   19.98 GiB
  DPO (2 models)             Actor(16P) + Ref(2P)                           9.99 GiB
  SimPO (1 model)            Actor(16P) only                                8.88 GiB
  GRPO + RLVR (no critic/RM) Actor(16P) + Ref(2P)                           9.99 GiB

Model-state VRAM bill for 7B dense model (7.00B params)
=========================================================================
  SFT / Pre-training         Actor(16P)                                   104.31 GiB
  PPO RLHF (4 models)        Actor(16P) + Critic(16P) + Ref(2P) + RM(2P)  234.69 GiB
  DPO (2 models)             Actor(16P) + Ref(2P)                         117.35 GiB
  SimPO (1 model)            Actor(16P) only                              104.31 GiB
  GRPO + RLVR (no critic/RM) Actor(16P) + Ref(2P)                         117.35 GiB
```

Look at the `PPO RLHF (4 models)` row. On Qwen3-0.6B, model state jumps from
**8.88 GiB to 19.98 GiB** ($2.25\times$). On a 7B model, model state jumps from
**104.31 GiB to 234.69 GiB**, nearly a quarter of a terabyte of VRAM before we
allocate a single KV cache block or activation tensor.

---

## The idea

### 1. Stage 1: The Bradley-Terry reward model

Given a prompt $x$ and two candidate completions $y_w$ and $y_l$ where $y_w$ won
and $y_l$ lost, we first need to train a scalar **Reward Model**
$r_\phi(x, y) \in \mathbb{R}$. We start from the SFT model $\pi^{\text{SFT}}$,
remove its $d \times V$ vocabulary head, and replace it with a single linear
projection $W_r \in \mathbb{R}^{1 \times d}$ that maps the final token's hidden
state $\mathbf{h}_{\text{last}} \in \mathbb{R}^d$ to a scalar score
$r_\phi(x, y) = W_r \mathbf{h}_{\text{last}}$.

Under the **Bradley-Terry preference model** (1952), the probability that
completion $y_w$ is preferred over $y_l$ depends only on the **difference** of
their scalar scores passed through the logistic sigmoid
$\sigma(z) = \frac{1}{1 + e^{-z}}$.

$$P_\phi(y_w \succ y_l \mid x) = \frac{e^{r_\phi(x, y_w)}}{e^{r_\phi(x, y_w)} + e^{r_\phi(x, y_l)}} = \frac{1}{1 + e^{-(r_\phi(x, y_w) - r_\phi(x, y_l))}} = \sigma\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)$$

We train $\phi$ by minimizing the negative log-likelihood over preference pairs
$(x, y_w, y_l) \sim \mathcal{D}$.

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Bigl[\log \sigma\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)\Bigr]$$

Check the numbers on two concrete pairs. When $r_\phi(x, y_w) = 2.0$ and
$r_\phi(x, y_l) = 0.0$, the model ranks the pair right by $+2.0$, so
$\sigma(2.0) \approx 0.8808$ and the loss is $-\log(0.8808) \approx 0.1269$.
When $r_\phi(x, y_w) = -1.0$ and $r_\phi(x, y_l) = +1.0$, the model ranks the
pair wrong by $-2.0$, so $\sigma(-2.0) \approx 0.1192$ and the loss jumps to
$-\log(0.1192) \approx 2.1269$.

Notice an algebraic invariance here that Lecture 38 (DPO) will use directly:
adding any prompt-dependent shift $c(x)$ to $r_\phi(x, y)$ leaves the difference
$r_\phi(x, y_w) + c(x) - r_\phi(x, y_l) - c(x)$ **unchanged**.

A Bradley-Terry preference RM emits a single score at `<eos>`, which works well
for chat style and tone. For multi-step math and coding reasoning, Cobbe et al.
(2021) and Lightman et al. (2023, *Let's Verify Step by Step*) introduced two
finer-grained reward architectures.

An **Outcome Reward Model (ORM)** takes a completion $s = (s_1, \dots, s_T)$
with a binary final correctness label $r \in \{0, 1\}$ and trains a per-token
scalar head $p_\phi(s_t) = \sigma(w_\phi(s_t))$ to predict that final outcome at
every token position $t$.

$$\mathcal{L}_{\text{ORM}}(\phi) = -\mathbb{E}_{(s, r)}\left[\frac{1}{T}\sum_{t=1}^T \Bigl(r \log p_\phi(s_t) + (1 - r)\log\bigl(1 - p_\phi(s_t)\bigr)\Bigr)\right]$$

A **Process Reward Model (PRM)** instead splits a chain-of-thought into $K$
reasoning steps $s_1, \dots, s_K$ separated by `\n\n` or `<step>` tokens and
trains a classifier $r_\phi(s_i \mid x, s_{<i})$ on each step's own correctness
label $y_{s_i} \in \{0, 1\}$.

$$\mathcal{L}_{\text{PRM}}(\phi) = -\mathbb{E}_{(x, s)}\left[\sum_{i=1}^K \Bigl(y_{s_i} \log r_\phi(s_i \mid x, s_{<i}) + (1 - y_{s_i})\log\bigl(1 - r_\phi(s_i \mid x, s_{<i})\bigr)\Bigr)\right]$$

Labeling every step by hand is expensive, so **Math-Shepherd** (Wang et al.,
2024) labels step $s_i$ automatically with Monte Carlo rollouts: it samples
$M = 8$ continuations from the prefix $(x, s_{\le i})$ and sets $y_{s_i}$ from
the fraction of continuations that reach the verified final answer $a^*$.

---

### 2. Stage 2: The KL-constrained RL objective

Once $r_\phi(x, y)$ is trained and frozen, we cannot simply optimize our policy
$\pi_\theta(y \mid x)$ to maximize $\mathbb{E}_{y \sim \pi_\theta}[r_\phi(x, y)]$
alone. The reward model $r_\phi$ is an imperfect neural proxy trained on a
finite dataset. If we push $\pi_\theta$ unrestrictedly toward maximum $r_\phi$,
the policy quickly finds out-of-distribution gibberish strings (such as
repeating `"the the the"` or sycophantic praise) that trick $r_\phi$ into
outputting $+50.0$. That failure mode is **reward hacking** (Goodhart's Law).

To keep $\pi_\theta$ near the region where $r_\phi$ is reliable, we subtract a
**KL-divergence leash** against the frozen SFT reference policy $\pi_{\text{ref}}$.

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\Bigl[r_\phi(x, y)\Bigr] - \beta \, \mathbb{D}_{\text{KL}}\bigl(\pi_\theta(y \mid x) \,\big\|\, \pi_{\text{ref}}(y \mid x)\bigr)$$

Here the sequence-level KL divergence sums the per-token log-probability ratios
along the generated sequence.

$$\mathbb{D}_{\text{KL}}\bigl(\pi_\theta \,\big\|\, \pi_{\text{ref}}\bigr) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right] = \mathbb{E}_{y \sim \pi_\theta}\left[\sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}\right]$$

---

### 3. Why PPO needs 4 models ($36\Psi$ bytes) and a per-token Critic

Walk through how **PPO** (Schulman et al., 2017) optimizes that objective. First,
we sample responses on-policy from the current snapshot
$\pi_{\theta_{\text{old}}}(y \mid x)$ using standard autoregressive decode
(Lecture 01). Second, we fold the KL penalty into per-token rewards: at every
token $t < T$, only the KL penalty fires, and at the final token $T$, we add the
scalar reward $r_\phi(x, y)$.

$$R_t = -\beta \log \frac{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} + \begin{cases} 0 & t < T \\ r_\phi(x, y) & t = T \end{cases}$$

Third, we need a per-token **Advantage** $\hat{A}_t$ that tells token $t$
whether choosing $y_t$ worked out better or worse than expected. Using the raw
future return $\sum_{k=t}^T R_k$ directly has huge variance across prompts and
token positions. So PPO trains a **second full transformer**, the **Critic
$V_\psi$**, to predict the expected future return from every partial prefix
$(x, y_{<t})$ via **Generalized Advantage Estimation (GAE, Schulman et al. 2015)**.

Define the 1-step temporal-difference residual
$\delta_t = R_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$ and write out the
$n$-step advantage estimators $\hat{A}_t^{(n)}$.

$$\begin{aligned}
\hat{A}_t^{(1)} &= R_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t) = \delta_t \\
\hat{A}_t^{(2)} &= R_t + \gamma R_{t+1} + \gamma^2 V_\psi(s_{t+2}) - V_\psi(s_t) = \delta_t + \gamma \delta_{t+1} \\
\hat{A}_t^{(n)} &= \sum_{l=0}^{n-1} \gamma^l \delta_{t+l}
\end{aligned}$$

GAE takes the exponentially weighted average of all $\hat{A}_t^{(n)}$ with
weights $(1 - \lambda)\lambda^{n-1}$. Using the geometric series
$\sum_{k=l}^\infty \lambda^k = \frac{\lambda^l}{1 - \lambda}$, the double sum
collapses into a single sum of discounted residuals.

$$\begin{aligned}
\hat{A}_t^{\text{GAE}(\gamma, \lambda)}
&= (1 - \lambda)\Bigl(\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(3)} + \cdots\Bigr) \\
&= (1 - \lambda)\Bigl(\delta_t(1 + \lambda + \lambda^2 + \cdots) + \gamma\delta_{t+1}(\lambda + \lambda^2 + \cdots) + \cdots\Bigr) \\
&= (1 - \lambda)\left(\delta_t \frac{1}{1-\lambda} + \gamma\delta_{t+1} \frac{\lambda}{1-\lambda} + \gamma^2\delta_{t+2} \frac{\lambda^2}{1-\lambda} + \cdots\right) \\
&= \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}
\end{aligned}$$

Fourth, we update the Actor $\pi_\theta$ using PPO's clipped surrogate
objective. Define the per-token importance ratio between the updated policy
$\pi_\theta$ and the rollout policy $\pi_{\theta_{\text{old}}}$.

$$\rho_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}$$

To stop a single gradient step from moving $\pi_\theta$ too far from
$\pi_{\theta_{\text{old}}}$, PPO clips $\rho_t(\theta)$ to
$[1 - \varepsilon, 1 + \varepsilon]$ (with $\varepsilon = 0.2$) and takes the
minimum of the unclipped and clipped terms.

$$\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_t\Bigl[\min\bigl(\rho_t(\theta)\hat{A}_t,\; \text{clip}(\rho_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\bigr)\Bigr]$$

??? question "Why does PPO take the `min` of the unclipped and clipped terms instead of clipping $\rho_t$ directly?"
    The sign of $\hat{A}_t$ flips what moving too far means. When $\hat{A}_t > 0$,
    the token was better than expected, so we want to **increase** $\rho_t$; the
    `min` caps the gain once $\rho_t > 1 + \varepsilon$ while still letting
    $\rho_t < 1 - \varepsilon$ recover. When $\hat{A}_t < 0$, the token was worse
    than expected, so we want to **decrease** $\rho_t$; now $\rho_t \hat{A}_t$ is
    negative, and taking the `min` caps the objective once $\rho_t < 1 - \varepsilon$
    so we do not drive $\pi_\theta(y_t)$ all the way to zero in one step.
    [Full answer](qa.md#why-does-ppo-take-the-min-of-unclipped-and-clipped-terms)

Now count the four models sitting in VRAM during a PPO step using Lecture 30's
ledger:

```
1. Actor (Policy)  pi_theta : trainable (BF16 weights + BF16 grads + FP32 AdamW) = 16 * Psi
2. Critic (Value)  V_psi    : trainable (BF16 weights + BF16 grads + FP32 AdamW) = 16 * Psi
3. Reference       pi_ref   : frozen    (BF16 weights only, forward-only)        =  2 * Psi
4. Reward Model    r_phi    : frozen    (BF16 weights only, forward-only)        =  2 * Psi
-------------------------------------------------------------------------------------------
Total PPO model-state memory                                                     = 36 * Psi
```

Beyond the $36\Psi$ VRAM bill sits a deeper problem: **the Critic $V_\psi$ takes
half of your training compute and memory ($16\Psi$), and its job is nearly
impossible**. On a 500-token math solution, the true accuracy reward arrives
only at token 500 ($0$ or $1$). At token 173, the Critic must somehow guess the
probability that a partial equation will end in a right answer 327 tokens later.
Whenever $V_\psi$ guesses poorly, every token's advantage $\hat{A}_t$ is noisy.

That bottleneck split modern post-training into two cleaner paths. In **Lecture
38 (DPO)**, we drop RL, the Reward Model, and the Critic altogether by solving
the KL-constrained objective in closed form ($36\Psi \to 18\Psi$). In **Lecture
39 (GRPO)**, we keep online RL, delete the $16\Psi$ Critic by comparing a group
of rollouts against their own mean, and replace the $2\Psi$ Reward Model with a
rule-based verifier ($36\Psi \to 16\Psi$ or $18\Psi$).

---

## The code

```python
import math


def bradley_terry_loss(r_chosen: float, r_rejected: float) -> float:
    diff = r_chosen - r_rejected
    return math.log1p(math.exp(-diff)) if diff >= 0 else -diff + math.log1p(math.exp(diff))


def ppo_actor_critic_bytes(params: int) -> int:
    # Actor (16P) + Critic (16P) + Reference (2P) + Reward Model (2P)
    return 36 * params


def ppo_clipped_surrogate(
    ratio: float, advantage: float, eps: float = 0.2
) -> float:
    clipped = min(max(ratio, 1.0 - eps), 1.0 + eps)
    return min(ratio * advantage, clipped * advantage)
```

---

## Build it

1. Implement `bradley_terry_loss`, `ppo_actor_critic_bytes`, and
   `ppo_clipped_surrogate` in `train/ppo_rlhf.py`.
2. Run the tests:

```bash
uv run pytest tests/test_37_ppo_rlhf.py -v
```

3. Verify both regimes of `ppo_clipped_surrogate`. When `advantage = +1.0` and
   `eps = 0.2`, raising `ratio` from `1.2` to `2.0` leaves the surrogate
   objective capped at `1.2`. When `advantage = -1.0` and `eps = 0.2`, lowering
   `ratio` from `0.8` to `0.1` leaves the surrogate objective capped at `-0.8`.

---

## What you should see

Bradley-Terry loss depends only on the score gap $r(x, y_w) - r(x, y_l)$, equals
$\ln 2 \approx 0.6931$ at a tie, and approaches $0$ as the margin grows
positive. Running full PPO requires **$36\Psi$ bytes of model state**
($2.25\times$ SFT), with $16\Psi$ spent on a Critic network that you throw away
as soon as training ends.

---

## Go deeper

- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022), the 3-stage SFT to RM to PPO pipeline.
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017), the clipped surrogate objective.
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) (Schulman et al., 2015), GAE($\lambda$).

---

## Check yourself

1. Why does adding $+100$ to both $r_\phi(x, y_w)$ and $r_\phi(x, y_l)$ leave the
   Bradley-Terry loss completely unchanged?
2. What happens if you maximize $\mathbb{E}_{y \sim \pi_\theta}[r_\phi(x, y)]$
   with $\beta = 0$ (no KL leash)?
3. Name all four models in PPO RLHF, state which are trainable vs frozen, and
   sum their byte counts to $36\Psi$.

---

## Next

**[38. Direct Preference Optimization (DPO) and reference-free variants](38-dpo-and-offline-preference.md)**:
how a 4-line algebraic change of variables cancels the intractable partition
function $Z(x)$ and collapses PPO's 4-model RL loop into a single binary
cross-entropy loss.

```bash
uv run python book/code/dpo_math.py
```

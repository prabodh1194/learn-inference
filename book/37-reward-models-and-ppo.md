# 37. Reward modeling and RLHF with PPO (the 4-model problem)

**Demo:** `book/code/rlhf_memory.py` · **Build:** `train/ppo_rlhf.py` · **Test:** `tests/test_37_ppo_rlhf.py`
**Moves:** aligns models to comparative preferences — and exposes the $36\Psi$ memory wall that motivated DPO and GRPO
**Prereq:** [36. SFT, LoRA, and QLoRA](36-sft-and-lora.md), [06. Sampling](06-sampling.md)

---

## The problem

Supervised fine-tuning (Lecture 36) has three structural ceilings:

1. **It has no negative signal.** Cross-entropy pushes up the probability of the
   target token $y_t^*$; it only pushes down other tokens in proportion to what
   the model already thought was likely. It cannot say *"this specific plausible
   hallucination or sycophantic answer is bad."*
2. **Writing demonstrations is harder than judging them.** Asking a human to
   write an optimal 500-line Python patch or poem from scratch is slow and
   caps the model at the annotator's skill; asking a human (or a test suite)
   *"which of these two model outputs is better: $y_w$ or $y_l$?"* is fast and
   scales.
3. **Exposure bias.** SFT trains only on ground-truth prefixes (**teacher
   forcing**). At inference time, the model conditions on its *own* sampled
   tokens.

The classic solution — **Reinforcement Learning from Human Feedback (RLHF)**
with **Proximal Policy Optimization (PPO)** (Ouyang et al., *InstructGPT*, 2022)
— works, and immediately creates a systems crisis: **it requires holding four
full language models in memory at the same time ($36\Psi$ bytes)** and training a
per-token **Critic (Value network)** whose job is nearly impossible when rewards
only arrive at the end of the response.

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

Look at the `PPO RLHF (4 models)` row:
- On Qwen3-0.6B, model state jumps from **8.88 GiB to 19.98 GiB** ($2.25\times$).
- On a 7B model, model state jumps from **104.31 GiB to 234.69 GiB** — nearly
  a quarter of a terabyte of VRAM before a single KV cache block or activation is
  allocated!

---

## The idea

### 1. Stage 1: The Bradley-Terry reward model

Given a prompt $x$ and two candidate completions $(y_w, y_l)$ where $y_w \succ y_l$
($y_w$ won, $y_l$ lost), how do we train a scalar **Reward Model**
$r_\phi(x, y) \in \mathbb{R}$?

Take the SFT model $\pi^{\text{SFT}}$, remove its $d \times V$ vocabulary head,
and replace it with a single linear projection $W_r \in \mathbb{R}^{1 \times d}$
that maps the final token's hidden state $\mathbf{h}_{\text{last}} \in \mathbb{R}^d$
to a scalar score $r_\phi(x, y) = W_r \mathbf{h}_{\text{last}}$.

Under the **Bradley-Terry preference model** (1952), the probability that
completion $y_w$ is preferred over $y_l$ depends only on the **difference** of
their scalar scores passed through the logistic sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$:

$$P_\phi(y_w \succ y_l \mid x) = \frac{e^{r_\phi(x, y_w)}}{e^{r_\phi(x, y_w)} + e^{r_\phi(x, y_l)}} = \frac{1}{1 + e^{-(r_\phi(x, y_w) - r_\phi(x, y_l))}} = \sigma\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)$$

Maximize the log-likelihood over preference pairs $(x, y_w, y_l) \sim \mathcal{D}$,
or equivalently minimize the negative log-likelihood:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\Bigl[\log \sigma\bigl(r_\phi(x, y_w) - r_\phi(x, y_l)\bigr)\Bigr]$$

Check the numbers on two concrete pairs:
- If $r_\phi(x, y_w) = 2.0$ and $r_\phi(x, y_l) = 0.0$ (correct ranking by $+2.0$):
  $$\sigma(2.0 - 0.0) = \frac{1}{1 + e^{-2}} \approx 0.8808 \implies \mathcal{L}_{\text{RM}} = -\log(0.8808) \approx 0.1269$$
- If $r_\phi(x, y_w) = -1.0$ and $r_\phi(x, y_l) = +1.0$ (wrong ranking by $-2.0$):
  $$\sigma(-2.0) = \frac{1}{1 + e^{2}} \approx 0.1192 \implies \mathcal{L}_{\text{RM}} = -\log(0.1192) \approx 2.1269$$

Notice an algebraic invariance here that Lecture 38 (DPO) will exploit: adding
any prompt-dependent constant $c(x)$ to $r_\phi(x, y)$ leaves the difference
$(r_\phi(x, y_w) + c(x)) - (r_\phi(x, y_l) + c(x))$ **completely unchanged**!

---

### 2. Stage 2: The KL-constrained RL objective

Once $r_\phi(x, y)$ is trained and frozen, why not just optimize our policy
$\pi_\theta(y \mid x)$ to maximize $\mathbb{E}_{y \sim \pi_\theta}[r_\phi(x, y)]$?

Because $r_\phi$ is an imperfect neural proxy trained on a finite dataset. Push
$\pi_\theta$ unrestrictedly toward maximum $r_\phi$ and it rapidly discovers
out-of-distribution gibberish strings (e.g., repeating `"the the the"` or
adversarial praise) that trick $r_\phi$ into outputting $+50.0$ — **reward
hacking** (Goodhart's Law).

To keep $\pi_\theta$ near the distribution where $r_\phi$ is valid, subtract a
**KL-divergence leash** against the frozen SFT reference policy $\pi_{\text{ref}}$:

$$\max_{\pi_\theta} \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot \mid x)}\Bigl[r_\phi(x, y)\Bigr] - \beta \, \mathbb{D}_{\text{KL}}\bigl(\pi_\theta(y \mid x) \,\big\|\, \pi_{\text{ref}}(y \mid x)\bigr)$$

where the sequence-level KL divergence is:

$$\mathbb{D}_{\text{KL}}\bigl(\pi_\theta \,\big\|\, \pi_{\text{ref}}\bigr) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right] = \mathbb{E}_{y \sim \pi_\theta}\left[\sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}\right]$$

---

### 3. Why PPO needs 4 models ($36\Psi$ bytes) and a per-token Critic

To optimize that objective with **PPO** (Schulman et al., 2017):

1. **Sample responses on-policy** from the current snapshot $\pi_{\theta_{\text{old}}}(y \mid x)$
   (ordinary autoregressive decode, Lecture 01!).
2. **Fold the KL penalty into per-token rewards:** at every token $t < T$, only
   the KL penalty fires; at the final token $T$, the scalar reward $r_\phi(x, y)$
   is added:
   $$R_t = -\beta \log \frac{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})} + \begin{cases} 0 & t < T \\ r_\phi(x, y) & t = T \end{cases}$$
3. **Estimate per-token Advantages $\hat{A}_t$ using a Critic (Value network)
   $V_\psi(x, y_{<t})$:**
   Why can't every token $t$ just use the raw return $\sum_{k=t}^T R_k$? Because
   the variance across prompts and token positions is enormous. So PPO trains a
   **second full transformer**, the **Critic $V_\psi$**, to predict the expected
   future return from every partial prefix $(x, y_{<t})$ via Generalized
   Advantage Estimation (GAE):
   $$\delta_t = R_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t), \qquad \hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$
4. **Update the Actor $\pi_\theta$ via the clipped surrogate objective:**
   Define the per-token importance ratio between the updated policy $\pi_\theta$
   and the rollout policy $\pi_{\theta_{\text{old}}}$:
   $$\rho_t(\theta) = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}$$
   To prevent a single gradient step from moving $\pi_\theta$ too far from
   $\pi_{\theta_{\text{old}}}$, PPO clips $\rho_t(\theta)$ to $[1 - \varepsilon, 1 + \varepsilon]$
   (typically $\varepsilon = 0.2$) and takes the pessimistic minimum:
   $$\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_t\Bigl[\min\bigl(\rho_t(\theta)\hat{A}_t,\; \text{clip}(\rho_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\bigr)\Bigr]$$

??? question "Why does PPO take the `min` of the unclipped and clipped terms instead of just clipping $\rho_t$ directly?"
    Because the sign of $\hat{A}_t$ flips what "too far" means!
    - When $\hat{A}_t > 0$ (the token was better than expected), you want to
      **increase** $\rho_t$, and the `min` caps the gain once $\rho_t > 1 + \varepsilon$
      (while still allowing $\rho_t < 1 - \varepsilon$ to recover if it dropped).
    - When $\hat{A}_t < 0$ (the token was worse than expected), you want to
      **decrease** $\rho_t$, so $\rho_t \hat{A}_t$ is negative; taking the `min`
      caps the reward once $\rho_t < 1 - \varepsilon$ so you don't drive $\pi_\theta(y_t)$
      all the way to zero in one step.
    [Full answer](qa.md#why-does-ppo-take-the-min-of-unclipped-and-clipped-terms)

Now count the four models sitting in VRAM during a PPO step, using Lecture 30's
ledger:

```
1. Actor (Policy)  pi_theta : trainable (BF16 weights + BF16 grads + FP32 AdamW) = 16 * Psi
2. Critic (Value)  V_psi    : trainable (BF16 weights + BF16 grads + FP32 AdamW) = 16 * Psi
3. Reference       pi_ref   : frozen    (BF16 weights only, forward-only)        =  2 * Psi
4. Reward Model    r_phi    : frozen    (BF16 weights only, forward-only)        =  2 * Psi
-------------------------------------------------------------------------------------------
Total PPO model-state memory                                                     = 36 * Psi
```

And notice the deeper flaw beyond the $36\Psi$ VRAM bill: **the Critic $V_\psi$
is half of your training compute and memory ($16\Psi$), and its job is nearly
impossible.** On a 500-token math solution, the true accuracy reward arrives only
at token $500$ ($0$ or $1$). The Critic must somehow guess, at token $173$, the
probability that the partial equation will end in a right answer 327 tokens later.
When $V_\psi$ is inaccurate, every token's advantage $\hat{A}_t$ is noisy.

That realization split the post-training field into two paths:
- **Path 1 (Lecture 38 — DPO):** eliminate RL, the Reward Model, **and** the
  Critic by solving the KL-constrained objective in closed form ($36\Psi \to 18\Psi$).
- **Path 2 (Lecture 39 — GRPO):** keep online RL, delete the Critic ($16\Psi$),
  and replace the Reward Model ($2\Psi$) with a rule-based verifier ($36\Psi \to 18\Psi$).

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

3. Verify both regimes of `ppo_clipped_surrogate`:
   - When `advantage = +1.0` and `eps = 0.2`, raising `ratio` from `1.2` to `2.0`
     does **not** increase the surrogate objective (stays capped at `1.2`).
   - When `advantage = -1.0` and `eps = 0.2`, lowering `ratio` from `0.8` to `0.1`
     does **not** increase the surrogate objective (stays capped at `-0.8`).

---

## What you should see

- **Bradley-Terry loss depends only on $r(x, y_w) - r(x, y_l)$**, equals $\ln 2 \approx 0.6931$
  at a tie, and approaches $0$ as the margin grows positive.
- **PPO requires $36\Psi$ bytes of model state** ($2.25\times$ SFT), with $16\Psi$
  spent on a Critic network you throw away the moment training ends.

---

## Go deeper

- **[Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)** (Ouyang et al., 2022) — the canonical 3-stage SFT $\to$ RM $\to$ PPO pipeline.
- **[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)** (Schulman et al., 2017) — the clipped surrogate objective.
- **[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)** (Schulman et al., 2015) — GAE($\lambda$).

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

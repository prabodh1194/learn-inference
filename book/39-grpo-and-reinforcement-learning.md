# 39. Group Relative Policy Optimization (GRPO) and verifiable rewards (RLVR)

**Demo:** `book/code/grpo_step.py` · **Build:** `train/grpo.py` · **Test:** `tests/test_39_grpo.py`
**Moves:** deletes PPO's $16\Psi$ Critic network; unlocks reasoning via rule-based verifiable rewards
**Prereq:** [37. Reward modeling and PPO](37-reward-models-and-ppo.md), [38. DPO](38-dpo-and-offline-preference.md), [28b. Reasoning and test-time compute](28b-reasoning-and-test-time-compute.md)

---

## The problem

In Lecture 37, we saw that PPO's single most expensive organ is the **Critic
(Value network) $V_\psi$**: a second trainable transformer of the same size as
the policy ($16\Psi$ bytes of state, plus its own forward and backward pass
every step) whose only job is to supply a **baseline** so the policy gradient
knows whether a completion was *better than expected*.

And on reasoning tasks (math proofs, code generation), the Critic's job is
worst of all: the true reward arrives only at the very end of a 4,000-token
trace, yet $V_\psi$ is asked to guess a value at every intermediate token.

Shao et al. (*DeepSeekMath*, 2024) asked a blunt question: **instead of training
a second neural network to guess the baseline for question $q$, why not just
sample a group of $G$ answers to question $q$ and use their own average reward
as the baseline?**

That single change is **Group Relative Policy Optimization (GRPO)** — the
algorithm behind DeepSeekMath and DeepSeek-R1.

---

## See it

```bash
uv run python book/code/grpo_step.py
```

```
GRPO group-relative advantage (G=4 completions for one question)
=========================================================================
  rewards   : [1.0, 0.0, 0.8, 0.2]
  mean      : 0.5000    population std : 0.4123
  answer 1  : reward= 1.0  ->  A_hat = (1.0 - 0.5) / 0.4123 =  +1.21  [REINFORCE (+)]
  answer 2  : reward= 0.0  ->  A_hat = (0.0 - 0.5) / 0.4123 =  -1.21  [SUPPRESS  (-)]
  answer 3  : reward= 0.8  ->  A_hat = (0.8 - 0.5) / 0.4123 =  +0.73  [REINFORCE (+)]
  answer 4  : reward= 0.2  ->  A_hat = (0.2 - 0.5) / 0.4123 =  -0.73  [SUPPRESS  (-)]

Per-token KL estimators: naive log(pi/ref) vs Schulman k3 = (ref/pi) - log(ref/pi) - 1
=========================================================================
  pi_theta    pi_ref    naive log(pi/ref)    k3 estimator (always >= 0)
      0.20      0.20              +0.0000                        0.0000
      0.40      0.20              +0.6931                        0.1931
      0.10      0.20              -0.6931                        0.3069
      0.02      0.20              -2.3026                        6.6974
```

Look at both parts of the output:
1. **Four rewards, one mean, one standard deviation** — and answers 1 and 3 get
   positive advantages (`+1.21`, `+0.73`) while answers 2 and 4 get negative
   advantages (`-1.21`, `-0.73`). An entire $16\Psi$ neural network just got
   replaced by 3 lines of arithmetic!
2. In the second table, when $\pi_\theta < \pi_{\text{ref}}$, the naive sample
   KL $\log(\pi_\theta / \pi_{\text{ref}})$ goes **negative** (`-0.6931`, `-2.3026`),
   whereas the **Schulman $k_3$ estimator** is **strictly non-negative on every
   single token** (`0.3069`, `6.6974`).

---

## The idea

### 1. Deriving the GRPO objective and group-relative advantage

For each question $q$, sample a group of $G$ completions $\{o_1, o_2, \dots, o_G\}$
from the current rollout policy $\pi_{\theta_{\text{old}}}(\cdot \mid q)$ and
score each completion with a reward function to obtain $\{r_1, r_2, \dots, r_G\}$.

Compute the group mean and standard deviation:

$$\mu_r = \frac{1}{G} \sum_{i=1}^G r_i, \qquad \sigma_r = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \mu_r)^2}$$

In outcome supervision, every token $t \in \{1, \dots, |o_i|\}$ inside completion
$o_i$ receives the **group-normalized z-score** as its advantage:

$$\hat{A}_{i,t} = \hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$

Check the demo's 4-completion example step by step ($r = [1.0, 0.0, 0.8, 0.2]$):

$$\mu_r = \frac{1.0 + 0.0 + 0.8 + 0.2}{4} = \frac{2.0}{4} = 0.5$$

$$\sigma_r^2 = \frac{(1.0 - 0.5)^2 + (0.0 - 0.5)^2 + (0.8 - 0.5)^2 + (0.2 - 0.5)^2}{4} = \frac{0.25 + 0.25 + 0.09 + 0.09}{4} = \frac{0.68}{4} = 0.17$$

$$\sigma_r = \sqrt{0.17} \approx 0.41231 \implies \hat{A}_1 = \frac{+0.5}{0.41231} \approx +1.2127, \quad \hat{A}_3 = \frac{+0.3}{0.41231} \approx +0.7276$$

Now plug $\hat{A}_{i,t}$ into the PPO clipped surrogate from Lecture 37, with
per-token importance ratio $\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$,
and add the KL penalty directly outside the surrogate:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{q,\, \{o_i\} \sim \pi_{\theta_{\text{old}}}}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Bigl(\min\bigl(\rho_{i,t}\hat{A}_{i,t},\, \text{clip}(\rho_{i,t}, 1-\varepsilon, 1+\varepsilon)\hat{A}_{i,t}\bigr) - \beta \, \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})\Bigr)\right]$$

```
PPO vs GRPO:
  PPO  :  sample 1 answer  ──► Critic V_psi (16*Psi network) ──► GAE(A_t)  ──► Clipped Surrogate
  GRPO :  sample G answers ──► (r_i - mean(r)) / std(r)      ──► A_i       ──► Clipped Surrogate - beta*KL
```

??? question "What happens in GRPO when all $G$ sampled completions for a question get the exact same reward (e.g., all $0$ on a hard problem or all $1$ on an easy one)?"
    Then $r_i - \mu_r = 0$ for every completion $i \in \{1, \dots, G\}$, so
    $\hat{A}_i = 0$ across the entire group! That prompt contributes **zero
    policy-gradient signal** on that step (only the $\beta\,\mathbb{D}_{\text{KL}}$
    term remains). This is an important practical property of GRPO: questions
    that are too easy ($100\%$ pass rate) or impossibly hard ($0\%$ pass rate
    within $G$ samples) automatically zero themselves out, concentrating gradient
    updates on the "frontier" problems where the model succeeds on some tries
    and fails on others. (DAPO's dynamic sampling explicitly filters out all-0
    and all-1 groups so rollout compute isn't wasted on zero-gradient steps.)
    [Full answer](qa.md#what-happens-in-grpo-when-all-g-samples-get-the-same-reward)

---

### 2. Why GRPO uses the Schulman $k_3$ KL estimator: proof of unbiasedness and non-negativity

Instead of folding $-\beta \log(\pi_\theta / \pi_{\text{ref}})$ into the reward
like PPO did, GRPO adds a per-token KL penalty using **John Schulman's $k_3$
estimator**. Define the inverse probability ratio at token $(i, t)$:

$$r = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}$$

Then the $k_3$ estimator is:

$$\mathbb{D}_{\text{KL}}^{(k_3)}\bigl(\pi_\theta \,\big\|\, \pi_{\text{ref}}\bigr) = r - \log r - 1 = \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1$$

Prove both properties of $k_3(r) = r - \log r - 1$ in four lines:

1. **Expectation equals the exact KL divergence (unbiased):**
   Take the expectation over $y \sim \pi_\theta$:
   $$\begin{aligned}
   \mathbb{E}_{y \sim \pi_\theta}\bigl[r - \log r - 1\bigr] &= \sum_y \pi_\theta(y) \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} - \sum_y \pi_\theta(y) \log \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} - \sum_y \pi_\theta(y) \\
   &= \underbrace{\sum_y \pi_{\text{ref}}(y)}_{= 1} + \sum_y \pi_\theta(y) \log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)} - \underbrace{1}_{= 1} \\
   &= \mathbb{D}_{\text{KL}}\bigl(\pi_\theta \,\big\|\, \pi_{\text{ref}}\bigr)
   \end{aligned}$$
2. **Strictly non-negative on every single sample ($k_3(r) \ge 0$):**
   By the tangent-line inequality for the concave logarithm function,
   $\log r \le r - 1$ for all $r > 0$, with equality if and only if $r = 1$
   ($\pi_\theta = \pi_{\text{ref}}$). Rearranging gives:
   $$r - \log r - 1 \ge 0 \quad \text{for every } r > 0$$

By contrast, the naive sample estimator $k_1 = -\log r = \log(\pi_\theta / \pi_{\text{ref}})$
is negative whenever $\pi_\theta(y) < \pi_{\text{ref}}(y)$, injecting high-variance
signed noise into per-token gradients, and its gradient at $\pi_\theta = \pi_{\text{ref}}$
is non-zero whereas $\nabla_\theta k_3\big|_{\pi_\theta = \pi_{\text{ref}}} = 0$!

---

### 3. RL with Verifiable Rewards (RLVR) and the length-normalization subtlety

In DeepSeek-R1-Zero, the neural Reward Model $r_\phi$ ($2\Psi$ bytes) is **also
deleted** and replaced by a deterministic, rule-based verifier (**RLVR**):

$$R_{\text{rule}}(q, o) = R_{\text{accuracy}}(q, o) + R_{\text{format}}(o)$$

- $R_{\text{accuracy}} \in \{0, 1\}$ checks whether the boxed math answer matches
  ground truth or whether the generated code passes the unit test suite.
- $R_{\text{format}}$ checks whether the reasoning trace sits inside `<think>...</think>`
  and the answer inside `<answer>...</answer>`.

Why does RLVR beat a neural reward model on reasoning? Because **you cannot
reward-hack a compiler or an exact math checker.** A neural reward model trained
on human preferences can be fooled by confident-looking algebra; a verifier only
pays when the final answer is actually right.

**One subtle trap in vanilla GRPO (Dr. GRPO / DAPO length bias):**
Look closely at the $\frac{1}{|o_i|}$ factor in front of $\sum_{t=1}^{|o_i|}$ in
the GRPO objective above:
- When $\hat{A}_i > 0$ (correct answer), dividing by $|o_i|$ gives a **larger**
  per-token gradient to **shorter** correct answers.
- When $\hat{A}_i < 0$ (wrong answer), dividing by $|o_i|$ gives a **smaller**
  per-token penalty to **longer** wrong answers — silently encouraging the model
  to ramble when it is wrong!
- **Fix (Dr. GRPO / DAPO):** divide by a constant `max_tokens` (or normalize over
  the total number of tokens across the group $\sum_{i=1}^G |o_i|$) instead of
  each completion's own length $|o_i|$, and mask out the $\sigma_r$ division when
  question-difficulty weighting is undesired.

---

### 4. The Unified Paradigm: SFT, RFT, DPO, PPO, and GRPO are one equation

DeepSeekMath (§4.2) shows that every post-training method in Part VII has the
**exact same gradient form**:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{(q, o) \sim \mathcal{D}}\left[\frac{1}{|o|} \sum_{t=1}^{|o|} \underbrace{GC(q, o, t)}_{\text{gradient coefficient}} \, \nabla_\theta \log \pi_\theta(o_t \mid q, o_{<t})\right]$$

They differ in only two choices: **where the data $\mathcal{D}$ comes from** and
**what scalar $GC(q, o, t)$ multiplies $\nabla_\theta \log \pi_\theta$**:

| Method | Data source $\mathcal{D}$ | Gradient coefficient $GC(q, o, t)$ | Models in VRAM |
|---|---|---|---|
| **SFT** (L36) | fixed human/teacher dataset | $1$ (trusts every token equally) | $1$ ($16\Psi$) |
| **RFT** (L40) | sampled from $\pi_\theta$, filtered by verifier | $\mathbb{I}[\text{correct}(q, o)]$ ($1$ if right, $0$ if wrong) | $1$ ($16\Psi$) |
| **DPO** (L38) | fixed preference pairs $(y_w, y_l)$ | $\pm \beta\,\sigma(\hat{r}_l - \hat{r}_w)$ | $2$ ($18\Psi$) |
| **PPO** (L37) | online rollout from $\pi_{\theta_{\text{old}}}$ | clipped GAE advantage from Critic $V_\psi$ | $4$ ($36\Psi$) |
| **GRPO** (L39) | online group rollout $\{o_1,\dots,o_G\} \sim \pi_{\theta_{\text{old}}}$ | clipped group z-score $\frac{r_i - \mu_r}{\sigma_r} - \beta\,\nabla k_3$ | $2$ ($18\Psi$) |

Notice the crucial difference between **RFT / SFT** and **GRPO**: RFT has
$GC \in \{0, 1\}$, so it only ever pushes **up** on winners ($GC = 1$) and
ignores losers ($GC = 0$). GRPO has **signed** advantages ($\hat{A}_i > 0$ on
winners, $\hat{A}_i < 0$ on losers), so it actively **pushes down** the mistakes
the current policy is making!

---

## The code

```python
import math


def group_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    n = len(rewards)
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    return [(r - mean) / (std + eps) for r in rewards]


def k3_kl_estimator(pi_prob: float, ref_prob: float) -> float:
    r = ref_prob / pi_prob
    return r - math.log(r) - 1.0


def grpo_token_loss(
    ratio: float,
    advantage: float,
    pi_prob: float,
    ref_prob: float,
    eps: float = 0.2,
    beta: float = 0.001,
) -> float:
    clipped = min(max(ratio, 1.0 - eps), 1.0 + eps)
    surrogate = min(ratio * advantage, clipped * advantage)
    kl = k3_kl_estimator(pi_prob, ref_prob)
    return -(surrogate - beta * kl)
```

---

## Build it

1. Implement `group_advantages`, `k3_kl_estimator`, and `grpo_token_loss` in
   `train/grpo.py`.
2. Run the tests:

```bash
uv run pytest tests/test_39_grpo.py -v
```

3. Verify that `sum(group_advantages(r))` is $0.0$ (up to floating-point rounding)
   for any reward list `r`, and that `k3_kl_estimator(pi_prob, ref_prob) >= 0.0`
   for all positive probabilities!

---

## What you should see

- **Group advantages always sum to zero** across the $G$ completions for a
  prompt: above-average completions are reinforced, below-average completions
  are suppressed.
- **The $k_3$ KL estimator equals $0$ when $\pi_\theta = \pi_{\text{ref}}$ and is
  strictly positive whenever $\pi_\theta \neq \pi_{\text{ref}}$, in either direction.**

---

## Go deeper

- **[DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)** (Shao et al., 2024) — Section 4.1 (GRPO) and Section 4.2 (Unified Paradigm).
- **[Approximating KL Divergence](http://joschu.net/blog/kl-approx.html)** (John Schulman, 2020) — why $k_3 = (q/p) - \log(q/p) - 1$ is the minimum-variance unbiased non-negative estimator.
- **[Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)](https://arxiv.org/abs/2503.20783)** (Liu et al., 2025) — the $\frac{1}{|o_i|}$ length-bias analysis.

---

## Check yourself

1. Compute the GRPO advantages $\hat{A}_1, \hat{A}_2, \hat{A}_3, \hat{A}_4$ by
   hand for group rewards $r = [1, 1, 0, 0]$.
2. Prove in three lines that $\mathbb{E}_{y \sim \pi_\theta}\bigl[\frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} - \log \frac{\pi_{\text{ref}}(y)}{\pi_\theta(y)} - 1\bigr] = \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$.
3. Why does dividing the token sum by $|o_i|$ penalize long wrong answers *less*
   per token than short wrong answers?

---

## Next

**[40. Reasoning pipelines (R1-Zero to R1) and knowledge distillation](40-reasoning-pipelines-and-distillation.md)**:
what happens when you point GRPO straight at a raw base model — and why
distilling a giant reasoner into a small model beats running RL on the small
model directly.

```bash
uv run python book/code/distillation_loss.py
```

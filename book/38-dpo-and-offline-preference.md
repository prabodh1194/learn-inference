# 38. Direct Preference Optimization (DPO) and reference-free variants

**Demo:** `book/code/dpo_math.py` · **Build:** `train/dpo.py` · **Test:** `tests/test_38_dpo.py`
**Moves:** preference alignment in $18\Psi$ bytes (2 models, 0 rollouts, 0 critic)
**Prereq:** [37. Reward modeling and PPO](37-reward-models-and-ppo.md)

---

## The problem

Lecture 37 left us with a $36\Psi$-byte, 4-model PPO pipeline that first fits a
reward model $r_\phi(x, y)$ to human preferences $(x, y_w, y_l)$ under the
Bradley-Terry loss, and then uses reinforcement learning to maximize the
KL-constrained objective:

$$\max_{\pi} \; \mathbb{E}_{y \sim \pi(\cdot \mid x)}\bigl[r(x, y)\bigr] - \beta \, \mathbb{D}_{\text{KL}}\bigl(\pi(y \mid x) \,\big\|\, \pi_{\text{ref}}(y \mid x)\bigr)$$

Rafailov et al. (*Direct Preference Optimization*, 2023) noticed a remarkable
mathematical fact: **that optimization problem has an exact closed-form
solution**. When we invert that solution to write the reward $r(x, y)$ in terms
of the optimal policy $\pi^*(y \mid x)$ and substitute it into the Bradley-Terry
model, the intractable normalizing constant $Z(x)$ **cancels out completely**.

We no longer need a separate reward model, a value critic, or an on-policy
sampling loop. We keep only the trainable policy $\pi_\theta$ ($16\Psi$) and the
frozen reference policy $\pi_{\text{ref}}$ ($2\Psi$), and we train $\pi_\theta$
with ordinary binary cross-entropy.

---

## See it

```bash
uv run python book/code/dpo_math.py
```

```
DPO implicit rewards, loss, and gradient weight (beta = 0.1)
=========================================================================
  case                              r_hat(w)   r_hat(l)   margin   grad_wt   loss
  Start of DPO (pi == pi_ref)          0.000      0.000    0.000     0.500   0.693
  Policy ranks WRONG (prefers y_l)    -0.200      0.400   -0.600     0.646   1.037
  Policy ranks RIGHT (prefers y_w)     0.400     -0.500    0.900     0.289   0.341
```

Look at the `grad_wt` column, which measures $\sigma(\hat{r}_l - \hat{r}_w)$. At
initialization, where $\pi_\theta = \pi_{\text{ref}}$, both implicit rewards are
$0.000$, the margin is $0.000$, the loss is $\ln 2 \approx 0.693$, and the
gradient weight is $0.500$. When the policy ranks the pair **wrong**
($\hat{r}_w = -0.200 < \hat{r}_l = +0.400$), the gradient weight jumps to
**0.646**, pushing harder on mistakes. When the policy already ranks the pair
**right** ($\hat{r}_w = +0.400 > \hat{r}_l = -0.500$), the gradient weight drops
to **0.289**, backing off once the preference is learned.

---

## The idea

### 1. Full step-by-step derivation of the DPO objective

Start from Lecture 37's KL-constrained reward maximization objective for a fixed
prompt $x$.

$$\max_{\pi} \; \sum_y \pi(y \mid x) \, r(x, y) - \beta \sum_y \pi(y \mid x) \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

In **Step 1**, we factor out $-\beta$ and fold $r(x, y)$ into a single
logarithm. Divide the objective by $\beta > 0$, which does not change the
maximizer $\pi^*$, and rewrite $r(x, y) / \beta = \log \exp\bigl(\frac{1}{\beta} r(x, y)\bigr)$.

$$\max_{\pi} \; \sum_y \pi(y \mid x) \left[\log \exp\left(\frac{1}{\beta} r(x, y)\right) - \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right]$$

Combine the two logarithms inside the brackets using $\log a - \log b = -\log(b / a)$.

$$= -\min_{\pi} \; \sum_y \pi(y \mid x) \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)}$$

In **Step 2**, we normalize the denominator into a valid probability
distribution. The denominator $\pi_{\text{ref}}(y \mid x) \exp\bigl(\frac{1}{\beta} r(x, y)\bigr)$
is positive for every $y$, and its sum over all possible completions
$y \in \mathcal{Y}$ is the **partition function** $Z(x)$.

$$Z(x) = \sum_{y \in \mathcal{Y}} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Multiply and divide inside the logarithm by $Z(x)$, and define the normalized
Gibbs distribution $\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\bigl(\frac{1}{\beta} r(x, y)\bigr)$.

$$\min_{\pi} \; \sum_y \pi(y \mid x) \left[\log \frac{\pi(y \mid x)}{\pi^*(y \mid x)} - \log Z(x)\right] = \mathbb{D}_{\text{KL}}\bigl(\pi(y \mid x) \,\big\|\, \pi^*(y \mid x)\bigr) - \log Z(x)$$

Because $\log Z(x)$ does not depend on $\pi$, and $\mathbb{D}_{\text{KL}}(\pi \,\|\, \pi^*) \ge 0$
with equality **if and only if $\pi = \pi^*$**, we obtain the unique global
optimum.

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

In **Step 3**, we invert that equation to solve for $r(x, y)$ in terms of
$\pi^*$. Divide both sides by $\pi_{\text{ref}}(y \mid x)$, multiply by $Z(x)$,
take the logarithm, and multiply by $\beta$.

$$\frac{\pi^*(y \mid x) \, Z(x)}{\pi_{\text{ref}}(y \mid x)} = \exp\left(\frac{1}{\beta} r(x, y)\right) \implies \boxed{r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)}$$

At first glance, $\log Z(x)$ looks fatal because computing $Z(x)$ requires
summing over every possible string $y \in \mathcal{Y}$ ($151{,}936^T$ strings).

In **Step 4**, we substitute that reward expression into the Bradley-Terry model
and watch $\beta \log Z(x)$ cancel. Recall from Lecture 37 that the
Bradley-Terry preference model depends only on the **difference**
$r(x, y_w) - r(x, y_l)$ for the **same prompt $x$**.

$$\begin{aligned}
r(x, y_w) - r(x, y_l) &= \left[\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \cancel{\beta \log Z(x)}\right] - \left[\beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \cancel{\beta \log Z(x)}\right] \\
&= \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\end{aligned}$$

Here $Z(x)$ vanishes because it depends only on $x$, never on $y_w$ or $y_l$.
Replace $\pi^*$ with our trainable policy $\pi_\theta$ and plug the difference
into the Bradley-Terry negative log-likelihood $-\log \sigma(r_w - r_l)$.

$$\boxed{\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]}$$

Define the **implicit reward** assigned by the language model itself.

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \beta \sum_{t=1}^{|y|} \bigl(\log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t})\bigr)$$

Then the DPO loss takes a clean one-line form.

$$\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\bigl(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)\bigr)$$

That identity explains why the paper is titled *"Your Language Model is Secretly
a Reward Model"*.

---

### 2. Deriving the DPO gradient: why naive unlikelihood collapses

Why do we need $\pi_{\text{ref}}$ and $\sigma(\cdot)$ at all instead of simply
minimizing $-\log \pi_\theta(y_w \mid x) + \log \pi_\theta(y_l \mid x)$?

Differentiate $\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma(u)$ where
$u = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$, using
$\frac{d}{du}[-\log \sigma(u)] = -(1 - \sigma(u)) = -\sigma(-u)$.

$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\beta \, \mathbb{E}_{(x, y_w, y_l)}\Biggl[\underbrace{\sigma\bigl(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\bigr)}_{\text{per-example wrongness weight}} \cdot \Bigl(\underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{push up } y_w} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{push down } y_l}\Bigr)\Biggr]$$

Look at the two factors inside the expectation. The vector difference
$\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)$
raises the log-probability of the winner $y_w$ and lowers the log-probability of
the loser $y_l$. Multiplying that vector sits the scalar weight
$\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)) \in (0, 1)$, which acts
as a **self-regulating brake**. Once $\hat{r}_\theta(x, y_w) \gg \hat{r}_\theta(x, y_l)$,
the weight $\sigma(\hat{r}_l - \hat{r}_w)$ approaches $0$ and the gradient turns
off. Without that brake and without $\pi_{\text{ref}}$ normalizing token
probabilities, pushing down $\log \pi_\theta(y_l \mid x)$ keeps firing forever
until common English words that appear in $y_l$ are driven to $-\infty$ and the
model collapses into gibberish.

---

### 3. Practical traps of DPO and modern variants (IPO, SimPO, DPO+NLL)

Four failure modes show up when we run DPO in practice, and each one produced a
widely used fix.

First comes **length bias**. Because $\log \pi(y \mid x) = \sum_{t=1}^{|y|} \log \pi(y_t \mid x, y_{<t})$
sums over all tokens in $y$, longer completions add more terms, so DPO can widen
the margin simply by favoring verbose outputs. **SimPO (Meng et al., 2024)**
replaces the reference-ratio sum with the **length-normalized average
log-probability** and a target margin $\gamma$.

$$\mathcal{L}_{\text{SimPO}}(\theta) = -\log \sigma\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma\right)$$

SimPO also drops $\pi_{\text{ref}}$ ($2\Psi$), bringing memory from $18\Psi$
back down to $16\Psi$ and cutting forward passes in half.

Second comes **deterministic-preference overfitting**. When $y_w$ beats $y_l$ on
$100\%$ of labels, $\sigma(u) \to 1$ only as $u \to +\infty$, so DPO can push
$\log(\pi_\theta / \pi_{\text{ref}})$ to huge values regardless of $\beta$.
**IPO (Azar et al., 2023)** replaces the logistic loss with a squared loss
$\bigl((\hat{r}_w - \hat{r}_l) - \frac{1}{2\beta}\bigr)^2$ that stops pushing as
soon as the margin reaches $\frac{1}{2\beta}$.

Third comes **preference displacement**, also called likelihood squeezing.
Because $\mathcal{L}_{\text{DPO}}$ depends only on the *difference*
$\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x)$, the optimizer can
widen the gap by **lowering the log-probability of both $y_w$ and $y_l$** as
long as $\log \pi_\theta(y_l \mid x)$ falls faster. That shifts probability mass
onto unseen tokens outside $(y_w, y_l)$ and hurts long-chain reasoning. **DPO+NLL**
(Pang et al., 2024; Llama 3, Qwen3) prevents that drop by adding a
length-normalized negative log-likelihood anchor on the winning completion.

$$\mathcal{L}_{\text{DPO+NLL}}(\theta) = \mathcal{L}_{\text{DPO}}(\theta) - \alpha \frac{\log \pi_\theta(y_w \mid x)}{|y_w|}$$

Similarly, **ORPO** penalizes the log odds-ratio
$\text{odds}_\theta(y) = \frac{\pi_\theta(y\mid x)}{1 - \pi_\theta(y\mid x)}$
alongside the SFT loss without needing $\pi_{\text{ref}}$.

Fourth comes **offline distribution shift**. DPO trains on a **fixed offline
dataset** $\mathcal{D}$. As $\pi_\theta$ moves away from the policy that
generated $\mathcal{D}$, it never sees its *own* fresh mistakes. **Online or
Iterative DPO** fixes this by sampling fresh pairs from $\pi_\theta$, scoring
them, and running another round, which leads directly to online RL in Lecture
39.

??? question "If DPO optimizes the exact same KL-constrained objective as PPO, why does online RL (GRPO/PPO) beat offline DPO on math and code?"
    The closed-form equivalence holds over the **full support of $\pi_\theta$**,
    whereas offline DPO only evaluates $\mathcal{L}_{\text{DPO}}$ on a fixed
    pre-recorded dataset $\mathcal{D}$. On math and coding tasks, as $\pi_\theta$
    updates, it explores new reasoning paths and new failure modes that never
    appeared in $\mathcal{D}$. Online RL samples from $\pi_\theta$ on every step
    and grades what the model is *doing right now*.
    [Full answer](qa.md#if-dpo-solves-the-same-equation-as-ppo-why-does-online-rl-beat-dpo-on-reasoning)

---

## The code

```python
import math


def implicit_reward(logp_pi: float, logp_ref: float, beta: float = 0.1) -> float:
    return beta * (logp_pi - logp_ref)


def dpo_loss(
    logp_pi_w: float,
    logp_pi_l: float,
    logp_ref_w: float,
    logp_ref_l: float,
    beta: float = 0.1,
) -> float:
    rw = implicit_reward(logp_pi_w, logp_ref_w, beta)
    rl = implicit_reward(logp_pi_l, logp_ref_l, beta)
    margin = rw - rl
    return math.log1p(math.exp(-margin)) if margin >= 0 else -margin + math.log1p(math.exp(margin))


def dpo_gradient_weight(
    logp_pi_w: float,
    logp_pi_l: float,
    logp_ref_w: float,
    logp_ref_l: float,
    beta: float = 0.1,
) -> float:
    rw = implicit_reward(logp_pi_w, logp_ref_w, beta)
    rl = implicit_reward(logp_pi_l, logp_ref_l, beta)
    diff = rl - rw
    return 1.0 / (1.0 + math.exp(-diff)) if diff >= 0 else math.exp(diff) / (1.0 + math.exp(diff))
```

---

## Build it

1. Implement `implicit_reward`, `dpo_loss`, and `dpo_gradient_weight` in
   `train/dpo.py`.
2. Run the tests:

```bash
uv run pytest tests/test_38_dpo.py -v
```

3. Verify that at initialization (`logp_pi_w == logp_ref_w` and
   `logp_pi_l == logp_ref_l`), `dpo_loss` equals $\ln 2 \approx 0.693147$ and
   `dpo_gradient_weight` equals $0.5$ regardless of what values `logp_ref_w` and
   `logp_ref_l` take.

---

## What you should see

At initialization ($\pi_\theta = \pi_{\text{ref}}$), every preference pair
starts at $\hat{r}_w = \hat{r}_l = 0$ and loss $\ln 2$, even when the reference
model already gives much higher raw probability to $y_w$ than to $y_l$. As
training proceeds, the gradient weight $\sigma(\hat{r}_l - \hat{r}_w)$ focuses
updates on misranked pairs and shrinks toward $0$ on pairs the model has already
learned.

---

## Go deeper

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023), Section 4 and Appendix A.1 to A.2 give the full derivation above.
- [A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)](https://arxiv.org/abs/2310.12036) (Azar et al., 2023).
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) (Meng et al., 2024).

---

## Check yourself

1. In $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$,
   why does $\beta \log Z(x)$ cancel when plugged into the Bradley-Terry model?
2. What would happen if you compared $y_w$ from prompt $x_1$ against $y_l$ from a
   *different* prompt $x_2$? Would $Z(x)$ still cancel?
3. Why is `dpo_loss` equal to $\ln 2$ at step 0 even when $\log \pi_{\text{ref}}(y_w \mid x) = -5$
   and $\log \pi_{\text{ref}}(y_l \mid x) = -50$?

---

## Next

**[39. Group Relative Policy Optimization (GRPO) and verifiable rewards (RLVR)](39-grpo-and-reinforcement-learning.md)**:
how DeepSeekMath and DeepSeek-R1 kept on-policy RL while deleting the $16\Psi$
Critic network entirely.

```bash
uv run python book/code/grpo_step.py
```

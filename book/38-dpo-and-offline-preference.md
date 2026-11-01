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
mathematical fact: **that optimization problem has an exact closed-form solution**,
and if you invert it to write the reward $r(x, y)$ in terms of the optimal
policy $\pi^*(y \mid x)$ and substitute that into the Bradley-Terry model, the
intractable normalizing constant $Z(x)$ **cancels out completely**.

No reward model. No critic. No on-policy sampling loop. Just your policy
$\pi_\theta$ ($16\Psi$) and the frozen reference policy $\pi_{\text{ref}}$ ($2\Psi$),
trained with ordinary binary cross-entropy.

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

Look at the `grad_wt` column ($\sigma(\hat{r}_l - \hat{r}_w)$):
- At initialization ($\pi_\theta = \pi_{\text{ref}}$), both implicit rewards are
  $0.000$, the margin is $0.000$, the loss is $\ln 2 \approx 0.693$, and the
  gradient weight is $0.500$.
- When the policy ranks the pair **wrong** ($\hat{r}_w = -0.200 < \hat{r}_l = +0.400$),
  the gradient weight jumps to **0.646** — pushing harder on mistakes.
- When the policy already ranks the pair **right** ($\hat{r}_w = +0.400 > \hat{r}_l = -0.500$),
  the gradient weight drops to **0.289** — backing off once the preference is learned.

---

## The idea

### 1. Full step-by-step derivation of the DPO objective

Start from Lecture 37's KL-constrained reward maximization objective for a fixed
prompt $x$:

$$\max_{\pi} \; \sum_y \pi(y \mid x) \, r(x, y) - \beta \sum_y \pi(y \mid x) \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

**Step 1: Factor out $-\beta$ and fold $r(x, y)$ into a single logarithm.**
Divide the objective by $\beta > 0$ (which does not change the maximizer $\pi^*$)
and use $r(x, y) / \beta = \log \exp\bigl(\frac{1}{\beta} r(x, y)\bigr)$:

$$\max_{\pi} \; \sum_y \pi(y \mid x) \left[\log \exp\left(\frac{1}{\beta} r(x, y)\right) - \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right]$$

Combine the two logarithms inside the bracket using $\log a - \log b = -\log(b / a)$:

$$= -\min_{\pi} \; \sum_y \pi(y \mid x) \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)}$$

**Step 2: Normalize the denominator into a valid probability distribution.**
The denominator $\pi_{\text{ref}}(y \mid x) \exp\bigl(\frac{1}{\beta} r(x, y)\bigr)$
is positive for every $y$, and its sum over all possible completions $y \in \mathcal{Y}$
is the **partition function** $Z(x)$:

$$Z(x) = \sum_{y \in \mathcal{Y}} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Multiply and divide inside the logarithm by $Z(x)$, and define the normalized
Gibbs distribution $\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\bigl(\frac{1}{\beta} r(x, y)\bigr)$:

$$\min_{\pi} \; \sum_y \pi(y \mid x) \left[\log \frac{\pi(y \mid x)}{\pi^*(y \mid x)} - \log Z(x)\right] = \mathbb{D}_{\text{KL}}\bigl(\pi(y \mid x) \,\big\|\, \pi^*(y \mid x)\bigr) - \log Z(x)$$

Because $\log Z(x)$ does not depend on $\pi$, and $\mathbb{D}_{\text{KL}}(\pi \,\|\, \pi^*) \ge 0$
with equality **if and only if $\pi = \pi^*$**, the unique global optimum is:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

**Step 3: Invert the equation to solve for $r(x, y)$ in terms of $\pi^*$.**
Divide both sides by $\pi_{\text{ref}}(y \mid x)$, multiply by $Z(x)$, take $\log$,
and multiply by $\beta$:

$$\frac{\pi^*(y \mid x) \, Z(x)}{\pi_{\text{ref}}(y \mid x)} = \exp\left(\frac{1}{\beta} r(x, y)\right) \implies \boxed{r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)}$$

At first glance, $\log Z(x)$ looks fatal: computing $Z(x)$ requires summing over
every possible string $y \in \mathcal{Y}$ ($151{,}936^T$ strings!).

**Step 4: Substitute into Bradley-Terry and watch $\beta \log Z(x)$ cancel!**
Recall from Lecture 37 that the Bradley-Terry preference model depends only on
the **difference** $r(x, y_w) - r(x, y_l)$ for the **same prompt $x$**:

$$\begin{aligned}
r(x, y_w) - r(x, y_l) &= \left[\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \cancel{\beta \log Z(x)}\right] - \left[\beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \cancel{\beta \log Z(x)}\right] \\
&= \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\end{aligned}$$

$Z(x)$ vanishes because it depends only on $x$, not on $y_w$ or $y_l$! Replace
$\pi^*$ with our trainable policy $\pi_\theta$ and plug the difference into the
Bradley-Terry negative log-likelihood $-\log \sigma(r_w - r_l)$:

$$\boxed{\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]}$$

Define the **implicit reward** assigned by the language model itself:

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} = \beta \sum_{t=1}^{|y|} \bigl(\log \pi_\theta(y_t \mid x, y_{<t}) - \log \pi_{\text{ref}}(y_t \mid x, y_{<t})\bigr)$$

Then the DPO loss is simply:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma\bigl(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)\bigr)$$

That is why the paper is titled *"Your Language Model is Secretly a Reward Model"*.

---

### 2. Deriving the DPO gradient: why naive unlikelihood collapses

Why not skip $\pi_{\text{ref}}$ and $\sigma(\cdot)$ and just minimize
$-\log \pi_\theta(y_w \mid x) + \log \pi_\theta(y_l \mid x)$?

Differentiate $\mathcal{L}_{\text{DPO}}(\theta) = -\log \sigma(u)$ where
$u = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$. Using the derivative
$\frac{d}{du}[-\log \sigma(u)] = -(1 - \sigma(u)) = -\sigma(-u)$:

$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\beta \, \mathbb{E}_{(x, y_w, y_l)}\Biggl[\underbrace{\sigma\bigl(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\bigr)}_{\text{per-example wrongness weight}} \cdot \Bigl(\underbrace{\nabla_\theta \log \pi_\theta(y_w \mid x)}_{\text{push up } y_w} - \underbrace{\nabla_\theta \log \pi_\theta(y_l \mid x)}_{\text{push down } y_l}\Bigr)\Biggr]$$

Look at the two factors inside the expectation:
1. $\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)$
   increases the log-probability of the winner $y_w$ and decreases the
   log-probability of the loser $y_l$.
2. $\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)) \in (0, 1)$ is a
   **self-regulating brake**: once $\hat{r}_\theta(x, y_w) \gg \hat{r}_\theta(x, y_l)$,
   $\sigma(\hat{r}_l - \hat{r}_w) \to 0$ and the gradient turns off! Without that
   brake (and without $\pi_{\text{ref}}$ normalizing token probabilities),
   pushing down $\log \pi_\theta(y_l \mid x)$ keeps firing forever until common
   English words that happen to appear in $y_l$ are driven to $-\infty$ and the
   model degenerates into gibberish.

---

### 3. Practical traps of DPO and modern variants (IPO, SimPO, Online DPO)

Three real failure modes show up when you run DPO in production:

1. **Length bias and SimPO (Meng et al., 2024).** Because $\log \pi(y \mid x) = \sum_{t=1}^{|y|} \log \pi(y_t \mid x, y_{<t})$
   is a **sum** over tokens, longer completions accumulate more terms, allowing
   DPO to exploit length differences between $y_w$ and $y_l$. **SimPO** replaces
   the reference-ratio sum with the **length-normalized average log-probability**
   plus a target margin $\gamma$:
   $$\mathcal{L}_{\text{SimPO}}(\theta) = -\log \sigma\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma\right)$$
   Notice a bonus systems win: SimPO deletes $\pi_{\text{ref}}$ ($2\Psi$),
   dropping memory from $18\Psi$ back to $16\Psi$ and cutting forward passes in half!
2. **Deterministic-preference overfitting and IPO (Azar et al., 2023).** If
   $y_w$ is preferred $100\%$ of the time in the dataset, $\sigma(u) \to 1$ only
   as $u \to +\infty$, so DPO can drive $\log(\pi_\theta / \pi_{\text{ref}})$ to
   huge values regardless of $\beta$. **IPO** replaces the logistic loss with a
   squared loss $\bigl((\hat{r}_w - \hat{r}_l) - \frac{1}{2\beta}\bigr)^2$ that
   stops pushing once the margin hits $\frac{1}{2\beta}$.
3. **Preference displacement (likelihood squeezing) and DPO+NLL (Pang et al., 2024).**
   Because $\mathcal{L}_{\text{DPO}}$ only depends on the *difference*
   $\log \pi_\theta(y_w \mid x) - \log \pi_\theta(y_l \mid x)$, the optimizer can
   widen the margin by **reducing the log-probability of both $y_w$ and $y_l$**
   as long as $\log \pi_\theta(y_l \mid x)$ falls faster! Where does the leaked
   probability mass go? Into out-of-distribution tokens never seen in $(y_w, y_l)$,
   which degrades reasoning and causes gibberish on long generations. Adding a
   length-normalized NLL anchor on the chosen response (**DPO+NLL** / **RPO**):
   $$\mathcal{L}_{\text{DPO+NLL}}(\theta) = \mathcal{L}_{\text{DPO}}(\theta) - \alpha \frac{\log \pi_\theta(y_w \mid x)}{|y_w|}$$
   or using **ORPO** (which penalizes the log odds-ratio $\text{odds}_\theta(y) = \frac{\pi_\theta(y\mid x)}{1 - \pi_\theta(y\mid x)}$
   directly alongside SFT loss without $\pi_{\text{ref}}$) prevents chosen likelihoods
   from collapsing.
4. **Offline distribution shift.** DPO trains on a **fixed offline dataset**
   $\mathcal{D}$. As $\pi_\theta$ moves away from the model that generated
   $\mathcal{D}$, it never sees its *own* fresh mistakes. **Online / Iterative
   DPO** fixes this by periodically sampling fresh pairs from $\pi_\theta$,
   scoring them, and running another round — which brings us right to the
   doorstep of online RL in Lecture 39.

??? question "If DPO optimizes the exact same KL-constrained objective as PPO, why does online RL (GRPO/PPO) beat offline DPO on math and code?"
    Because the equivalence theorem holds over the **full support of $\pi_\theta$**,
    whereas offline DPO only evaluates $\mathcal{L}_{\text{DPO}}$ on a fixed
    pre-recorded dataset $\mathcal{D}$. On math and coding tasks, as $\pi_\theta$
    changes, it invents new reasoning paths and new failure modes that never
    appeared in $\mathcal{D}$. Online RL samples from $\pi_\theta$ every step and
    grades what the model is *actually doing right now*.
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
   `logp_ref_l` take!

---

## What you should see

- **At initialization ($\pi_\theta = \pi_{\text{ref}}$), every preference pair starts
  at $\hat{r}_w = \hat{r}_l = 0$ and loss $\ln 2$**, even if the reference model
  already assigns much higher raw probability to $y_w$ than $y_l$.
- **The gradient weight $\sigma(\hat{r}_l - \hat{r}_w)$ automatically prioritizes
  misranked pairs** and decays toward $0$ on already-learned pairs.

---

## Go deeper

- **[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)** (Rafailov et al., 2023) — Section 4 and Appendix A.1–A.2 contain the full derivation above.
- **[A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)](https://arxiv.org/abs/2310.12036)** (Azar et al., 2023).
- **[SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)** (Meng et al., 2024).

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

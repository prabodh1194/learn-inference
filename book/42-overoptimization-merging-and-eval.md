# 42. Over-optimization, model merging, and post-training evaluation

**Demo:** `book/code/merging_and_eval.py` · **Test:** `tests/test_42_merging_and_eval.py`
**Moves:** locates the exact KL peak $d^* = \frac{\alpha}{2\beta}$ before reward hacking sets in, merges specialist checkpoints in weight space without retraining, and separates `pass@k` from `pass^k`
**Prereq:** [37. Reward modeling and PPO](37-reward-models-and-ppo.md), [39. GRPO](39-grpo-and-reinforcement-learning.md), [41. RL infrastructure](41-rl-systems-inference-meets-training.md)

---

## The problem

In Lectures 36 to 41, we built the full post-training stack: SFT, Rejection
Sampling, PPO, DPO, GRPO, Distillation, and RL infrastructure. Three final
questions separate a lab prototype from a production release (`Tülu 3`,
`OLMo 3`, `Qwen3`, `Claude`).

First, when you optimize a policy against a learned reward model, how far can
you push KL divergence before the model starts gaming the reward model and
getting worse in the real world? Second, when you train separate specialist
checkpoints for math, coding, safety, and chat, how can you combine them into
one model without paying for another training run? Third, when you evaluate the
finished model on math or tool-use benchmarks, how do you avoid being fooled by
sampling noise, lucky retries (`pass@k` vs. `pass^k`), or test-set leakage?

---

## See it

```bash
uv run python book/code/merging_and_eval.py
```

```text
Gao et al. Over-Optimization Law (alpha=1.0, beta=0.25 -> peak d*=2.0)
------------------------------------------------------------------------
  d = sqrt(KL) = 0.0 (KL= 0.0) | R_proxy = +0.00 | R_gold = +0.00
  d = sqrt(KL) = 1.0 (KL= 1.0) | R_proxy = +1.00 | R_gold = +0.75
  d = sqrt(KL) = 2.0 (KL= 4.0) | R_proxy = +2.00 | R_gold = +1.00
  d = sqrt(KL) = 3.0 (KL= 9.0) | R_proxy = +3.00 | R_gold = +0.75
  d = sqrt(KL) = 4.0 (KL=16.0) | R_proxy = +4.00 | R_gold = +0.00

TIES-Merging on 3 task vectors: [0.5, -0.5, 0.30000000000000004]
n=10, c=8 (80% base): pass@5 = 1.0000 vs pass^5 = 0.3277
```

Look at the three results. In the top table, the proxy reward $R_{\text{proxy}}$
keeps climbing forever (`+1.00 -> +2.00 -> +4.00`), while the true gold reward
$R_{\text{gold}}$ peaks at $d^* = 2.0$ ($\text{KL} = 4.0\text{ nats}$) and falls
 all the way back to `+0.00` by $d = 4.0$. In the bottom line, a model that
succeeds 8 times out of 10 scores **100.0% on `pass@5`**, yet only **32.8% on
`pass^5`**.

---

## The idea

### 1. Quantitative over-optimization: Goodhart's Law in KL space

When we optimize a policy $\pi_\theta$ against a learned proxy reward model
$r_{\text{proxy}}(x, y)$, **Goodhart's Law** warns that once a proxy score
becomes an optimization target, it stops being a reliable measure.

Gao et al. (2023) measured this gap across model scales by plotting both the
proxy reward gain $R_{\text{proxy}}$ and the true held-out gold reward gain
$R_{\text{gold}}$ as a function of the **square-root KL distance** from the
reference policy:

$$d = \sqrt{\mathbb{D}_{\text{KL}}\bigl(\pi_\theta \,\big\|\, \pi_{\text{ref}}\bigr)}$$

Across both Best-of-$N$ and RL runs, the proxy reward climbs linearly with $d$,
while the true gold reward follows a concave quadratic curve (or
$d(\alpha - \beta \ln d)$ under Best-of-$N$):

$$R_{\text{proxy}}(d) = \alpha d, \qquad R_{\text{gold}}(d) = d(\alpha - \beta d) = \alpha d - \beta d^2$$

Here $\alpha > 0$ is the initial alignment slope and $\beta > 0$ measures how
easily the proxy reward model can be hacked (larger reward models have smaller
$\beta$).

We can find the exact KL budget where true quality peaks by differentiating
$R_{\text{gold}}(d)$ with respect to $d$ and setting the derivative to zero.

$$\frac{d R_{\text{gold}}}{d d} = \alpha - 2\beta d = 0 \implies d^* = \frac{\alpha}{2\beta}, \qquad \mathbb{D}_{\text{KL}}^* = (d^*)^2 = \frac{\alpha^2}{4\beta^2}$$

Now plug $d^* = \frac{\alpha}{2\beta}$ back into both reward formulas to compare
what the true gold judge sees against what your proxy reward model reports at
the peak.

$$R_{\text{gold}}(d^*) = \alpha\left(\frac{\alpha}{2\beta}\right) - \beta\left(\frac{\alpha}{2\beta}\right)^2 = \frac{\alpha^2}{2\beta} - \frac{\alpha^2}{4\beta} = \frac{\alpha^2}{4\beta}$$

$$R_{\text{proxy}}(d^*) = \alpha\left(\frac{\alpha}{2\beta}\right) = \frac{\alpha^2}{2\beta} = 2 \, R_{\text{gold}}(d^*)$$

Even if you stop at the **exact optimal KL distance $d^*$**, half of the proxy
reward gain you see on your training dashboard is already an illusion. And if
you keep training past $d = 2 d^* = \frac{\alpha}{\beta}$, $R_{\text{gold}}(d)$
turns negative: the over-optimized RL policy is now worse than the starting SFT
checkpoint $\pi_{\text{ref}}$. In practice, that over-optimization shows up as
**sycophancy** (flattering the user's wrong beliefs), padded verbosity, or
**over-refusal** (refusing harmless requests like *"how do I kill a Linux
process"*).

---

### 2. Persona vectors and Assistant Axis activation capping

How can we stop a model's persona from drifting into sycophancy or roleplay
delusion during a long multi-turn conversation without retraining its weights?
Recent mechanistic interpretability work (Chen et al., 2025; Lu et al., 2026)
steers behavior directly inside the residual stream
$\mathbf{h}_\ell \in \mathbb{R}^d$ at layer $\ell$.

To find the direction in activation space that encodes a trait, we collect a set
of responses $S^+$ that show the trait and a set of responses $S^-$ that do not.
The **persona vector** $\mathbf{v}_\ell$ is the difference between their mean
residual activations:

$$\mathbf{v}_\ell = \frac{1}{|S^+|}\sum_{i \in S^+} \mathbf{a}_\ell^{(i)} - \frac{1}{|S^-|}\sum_{j \in S^-} \mathbf{a}_\ell^{(j)}$$

When Lu et al. (2026) extracted persona vectors across $275+$ character
archetypes, they found that the first principal component of persona space lines
up with a single unit direction $\mathbf{v}$ ($\|\mathbf{v}\|_2 = 1$) pointing
toward the analytical, helpful **Assistant** persona. In long or emotionally
charged chats, the residual stream's projection $p = \langle \mathbf{h}, \mathbf{v} \rangle$
onto this **Assistant Axis** can slide below a safe threshold $\tau$, causing
the model to abandon its assistant persona.

**Activation Capping** prevents that drift at inference time by applying a
one-sided clamp along $\mathbf{v}$.

$$\mathbf{h}' = \mathbf{h} - \mathbf{v} \cdot \min\bigl(\langle \mathbf{h}, \mathbf{v} \rangle - \tau,\; 0\bigr)$$

Check both regimes of that formula. When the activation is already in the **safe
region** ($p = \langle \mathbf{h}, \mathbf{v} \rangle \ge \tau$), we have
$\min(p - \tau, 0) = 0$, so $\mathbf{h}' = \mathbf{h}$ and normal prompts are
completely untouched. When the activation enters the **drifted region**
($p = \langle \mathbf{h}, \mathbf{v} \rangle < \tau$), we have
$\min(p - \tau, 0) = p - \tau$, so $\mathbf{h}' = \mathbf{h} - (p - \tau)\mathbf{v}$.
Take the inner product of $\mathbf{h}'$ with the unit vector $\mathbf{v}$
(using $\langle \mathbf{v}, \mathbf{v} \rangle = 1$):

$$\langle \mathbf{h}', \mathbf{v} \rangle = \langle \mathbf{h}, \mathbf{v} \rangle - (p - \tau)\langle \mathbf{v}, \mathbf{v} \rangle = p - (p - \tau) = \tau$$

The projection along the Assistant Axis is lifted right back to the floor $\tau$,
while every feature direction $\mathbf{u}$ orthogonal to $\mathbf{v}$
($\langle \mathbf{u}, \mathbf{v} \rangle = 0$) stays unchanged.

---

### 3. Model merging: Task Arithmetic, DARE, and TIES-Merging

Post-training teams (`Tülu 3`, `OLMo 3`, `Command A`, `Qwen3`) often train
several specialist checkpoints $\theta_1, \dots, \theta_M$ (one for math, one
for coding, one for safety, or several hyperparameter seeds) starting from the
**same base model** $\theta_{\text{base}}$. Because all $M$ checkpoints sit in
the same loss basin around $\theta_{\text{base}}$, we can combine their skills
directly in weight space without running another GPU step.

For each fine-tuned model $m$, define its **task vector** (or weight delta) as
the shift from the base checkpoint:

$$\tau_m = \theta_m - \theta_{\text{base}}$$

**Linear Task Arithmetic (Ilharco et al., 2023)** simply adds a weighted sum of
the task vectors back onto $\theta_{\text{base}}$:

$$\theta_{\text{merged}} = \theta_{\text{base}} + \sum_{m=1}^M \lambda_m \tau_m$$

Why does plain linear addition lose accuracy when we merge several specialists?
Two task vectors $\tau_1$ and $\tau_2$ often modify the same weight coordinate
$i$ with **redundant** magnitudes (blowing up the weight) or **opposing signs**
($\tau_{1,i} = +0.6$ and $\tau_{2,i} = -0.6$ cancel to $0$ and erase both models'
updates).

**DARE (Drop And REscale, Yu et al. 2024)** reduces parameter collisions by
randomly zeroing out a fraction $p$ (often $p = 0.90$) of the delta coordinates
in each $\tau_m$ and rescaling the survivors by $\frac{1}{1 - p}$, just like
inverted dropout.

$$m_i \sim \text{Bernoulli}(1 - p), \qquad \tilde{\tau}_{m,i} = \frac{m_i}{1 - p}\,\tau_{m,i}$$

Why does dropping $90\%$ of the fine-tuning delta leave the model's behavior
intact? Take the expectation of $\tilde{\tau}_{m,i}$:

$$\mathbb{E}[\tilde{\tau}_{m,i}] = \Pr[m_i = 1] \cdot \frac{\tau_{m,i}}{1 - p} + \Pr[m_i = 0] \cdot 0 = (1 - p)\frac{\tau_{m,i}}{1 - p} = \tau_{m,i}$$

Because each linear layer computes $(\theta_{\text{base}} + \tilde{\tau}_m) x$
as a sum over thousands of input channels, keeping the expectation of every
coordinate preserves the expected layer output while cutting two-model overlap
collisions by $(1 - p)^2$. At $p = 0.9$, only $0.1^2 = 1\%$ of coordinates
collide between two merged models.

**TIES-Merging (Trim, Elect Sign, Disjoint Merge, Yadav et al. 2023)** resolves
both magnitude noise and sign conflicts in three deterministic steps. In **Step
1 (Trim)**, we keep only the top-$k\%$ largest-magnitude coordinates in each task
vector $\tau_m$ and set the small noisy shifts to $0$ to form $\hat{\tau}_m$. In
**Step 2 (Elect Sign)**, for each parameter coordinate $i$, we sum the trimmed
shifts over all $M$ models and take the sign of that total mass as the winning
sign $\gamma_i$.

$$\gamma_i = \text{sign}\!\left(\sum_{m=1}^M \hat{\tau}_{m,i}\right)$$

In **Step 3 (Disjoint Merge)**, we average **only** the models whose non-zero
shift at coordinate $i$ agrees with the winning sign $\gamma_i$.

$$\tau_i^{\text{TIES}} = \frac{1}{|\mathcal{A}_i|} \sum_{m \in \mathcal{A}_i} \hat{\tau}_{m,i}, \qquad \mathcal{A}_i = \{\, m \mid \text{sign}(\hat{\tau}_{m,i}) = \gamma_i \,\}$$

Check the demo's 3-model example on coordinate $0$, where the trimmed shifts are
$\hat{\tau}_{\cdot, 0} = [+0.6, +0.4, -0.2]$. A naive average gives
$(0.6 + 0.4 - 0.2)/3 = 0.2667$, diluted by model 3's opposing sign. Under TIES,
the sum is $+0.8 > 0$, so the winning sign is $\gamma_0 = +1$. Only models 1 and
2 agree ($\mathcal{A}_0 = \{1, 2\}$), and their disjoint average is
$(0.6 + 0.4)/2 = \mathbf{0.5000}$.

---

### 4. Post-training evaluation: `pass@k` vs. `pass^k`, variance, and decontamination

Once a model is trained and merged, three evaluation traps can misreport its
real capability.

First, distinguish **discovery (`pass@k`)** from **reliability (`pass^k`)**.
Suppose you sample $n \ge k$ independent completions on a coding or math problem
and $c$ of those $n$ completions pass the verifier.

The metric **`pass@k` (Chen et al., 2021)** asks: if we draw a random subset of
$k$ completions without replacement, what is the probability that **at least
one** of the $k$ completions is right? That equals $1$ minus the probability
that all $k$ draws come from the $n - c$ wrong completions.

$$\text{pass@}k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}} = 1 - \prod_{i=0}^{k-1} \frac{n - c - i}{n - i}$$

By contrast, for production tool-use agents (`\tau$-bench`, Yao et al. 2024),
there is no oracle verifier at serving time to discard failed runs after an
agent has already booked the wrong flight or edited a customer record. A user
who runs the agent $k$ times needs **every single one of the $k$ trials to
succeed**, which is measured by **`pass^k`**:

$$\text{pass}^k = \left(\frac{c}{n}\right)^k$$

Look at how fast those two curves diverge when a model has an $80\%$ single-try
success rate ($n = 10, c = 8$). Because there are only $n - c = 2$ wrong samples,
any draw of $k = 5$ samples without replacement is guaranteed to contain at
least 3 right answers, so `pass@5` is **$1.0000$ ($100\%$)**. Yet the chance
that the agent succeeds on 5 customer requests in a row is only
`pass^5` $= 0.80^5 = \mathbf{0.3277}$ (**$32.8\%$**).

??? question "Why is pass@k the wrong metric for production tool-use agents (and why pass^k is used instead)?"
    `pass@k` measures whether **at least one** of $k$ attempts succeeds, which
    only helps when you have a test suite at serving time to throw away the
    $k-1$ failures. Production agents modifying databases or external APIs have
    no oracle undo button: a customer running the agent $k$ times needs **every**
    trial to succeed, which happens with probability $\text{pass}^k = (c/n)^k$.
    [Full answer](qa.md#why-is-passk-the-wrong-metric-for-production-tool-use-agents-and-why-passk-is-used-instead)

Second, watch out for **small-benchmark sampling variance (`avg@K`)**. AIME has
only $30$ questions. Evaluating a reasoning model once at temperature $0.6$ has
a run-to-run standard deviation of roughly $\sigma_1 \approx 3.0$ percentage
points, which is larger than the gap between many competing algorithms. Sampling
$K = 32$ or $64$ completions per question and reporting the mean accuracy
(`avg@32` or `avg@64`, as in `OLMo 3` and `DeepSeek-R1`) shrinks the standard
error by $\sigma_1 / \sqrt{K}$ down to $\approx 0.4$ to $0.5$ points.

Third, enforce **decontamination** before training begins. Because synthetic SFT
and RLVR prompt pools scrape math forums and coding sites, benchmark problems
from `MATH-500`, `GPQA`, `HumanEval`, or `TruthfulQA` frequently leak into raw
prompt sets. Pipelines like `Tülu 3` strip any training prompt that shares an
8-gram token sequence (or a 50-character normalized substring) with any held-out
evaluation split.

---

## What you should see

Running `book/code/merging_and_eval.py` shows that under Gao et al.'s scaling
law, half of the proxy reward gain at the optimal stopping point $d^* = \frac{\alpha}{2\beta}$
is already reward hacking, and pushing past $2d^*$ makes the model worse than the
initial SFT checkpoint. In weight space, TIES-Merging prevents opposing-sign
task vectors from canceling out dominant specialist updates, while `pass^k`
exposes how an $80\%$-accurate agent fails two-thirds of the time over 5
consecutive runs.

---

## Go deeper

- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) (Gao et al., 2023), the $R_{\text{gold}}(d) = \alpha d - \beta d^2$ empirical law.
- [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) (Yadav et al., 2023) and [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE)](https://arxiv.org/abs/2311.03099) (Yu et al., 2024).
- [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (Chen et al., 2021), the unbiased `pass@k` estimator, and [$\tau$-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/abs/2406.12045) (Yao et al., 2024), the `pass^k` reliability metric.

---

## Check yourself

1. Given $R_{\text{proxy}}(d) = \alpha d$ and $R_{\text{gold}}(d) = \alpha d - \beta d^2$,
   find the KL distance $\mathbb{D}_{\text{KL}} = d^2$ where $R_{\text{gold}}$
   peaks and the distance where $R_{\text{gold}}$ falls back to zero.
2. Why does rescaling surviving delta weights by $\frac{1}{1 - p}$ in DARE
   preserve the expected output of a linear layer when $90\%$ of delta weights
   are dropped at random?
3. A coding agent passes 9 out of 10 trials on a benchmark task ($n = 10, c = 9$).
   Compute both `pass@2` and `pass^5`.

---

**Part VII, and the full course, complete.** You now hold both halves of the
modern LLM stack in one mathematical model: how tokens are trained into weights
(Parts VI and VII), how weights are served back into tokens (Parts I to V), and
why modern RL post-training is where both halves fuse into a single loop.

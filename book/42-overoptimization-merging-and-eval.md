# Lecture 42: Over-Optimization, Model Merging & Post-Training Evaluation

In Lectures 36–41, we built the complete post-training stack: SFT, Rejection Sampling, PPO, DPO, GRPO, Distillation, and RL Systems. Three final questions separate a research prototype from a frontier release (`Tülu 3`, `OLMo 3`, `Qwen3`, `Claude`):

1. **How far can you push RL against a proxy reward before the model degrades (Over-Optimization & Persona Drift)?**
2. **How do you combine multiple specialist checkpoints (math, coding, safety, chat) without retraining (Model Merging: Task Arithmetic, TIES, DARE)?**
3. **How do you measure reasoning and agentic models without being fooled by sampling noise, formatting mismatches, or benchmark contamination?**

---

### 1. Quantitative Over-Optimization (Goodhart's Law in KL Space)

When optimizing a policy $\pi_\theta$ against a learned proxy reward model $r_{\text{proxy}}(x, y)$, **Goodhart's Law** states that once a proxy measure becomes an optimization target, it ceases to be a valid measure.

Gao et al. (2023) measured this effect across model sizes by plotting both the proxy reward score $R_{\text{proxy}}$ and the true held-out gold reward score $R_{\text{gold}}$ as functions of the **square-root KL optimization distance** from the reference policy:

$$d = \sqrt{\mathbb{D}_{\text{KL}}\bigl(\pi_\theta \,\big\|\, \pi_{\text{ref}}\bigr)}$$

Empirically, the proxy score climbs linearly with $d$, while the true gold reward follows a concave quadratic law (or $d(\alpha - \beta \ln d)$ under BoN):

$$R_{\text{proxy}}(d) = \alpha d, \qquad R_{\text{gold}}(d) = d(\alpha - \beta d) = \alpha d - \beta d^2$$

where $\alpha > 0$ measures initial alignment slope and $\beta > 0$ (which shrinks as reward-model parameter count $N_{\text{RM}}$ grows) measures reward-hacking susceptibility.

#### Deriving the Optimal KL Budget $d^*$ and the Zero-Return Point

1. **Find the peak gold reward $d^*$:** Differentiate $R_{\text{gold}}(d)$ with respect to $d$ and set the derivative to zero:
   $$\frac{d R_{\text{gold}}}{d d} = \alpha - 2\beta d = 0 \implies d^* = \frac{\alpha}{2\beta}, \qquad \mathbb{D}_{\text{KL}}^* = (d^*)^2 = \frac{\alpha^2}{4\beta^2}$$
2. **Compare gold vs. proxy reward at the peak $d^*$:**
   $$R_{\text{gold}}(d^*) = \alpha\left(\frac{\alpha}{2\beta}\right) - \beta\left(\frac{\alpha}{2\beta}\right)^2 = \frac{\alpha^2}{2\beta} - \frac{\alpha^2}{4\beta} = \frac{\alpha^2}{4\beta}$$
   $$R_{\text{proxy}}(d^*) = \alpha\left(\frac{\alpha}{2\beta}\right) = \frac{\alpha^2}{2\beta} = 2 \, R_{\text{gold}}(d^*)$$
   Even at the **exact optimal stopping point $d^*$**, half of the apparent proxy reward gain is already illusory reward hacking!
3. **Past $d > 2 d^* = \frac{\alpha}{\beta}$:** $R_{\text{gold}}(d) < 0$ — the over-optimized RL policy is worse than the starting SFT checkpoint $\pi_{\text{ref}}$, exhibiting **sycophancy** (telling the user whatever they want to hear), extreme verbosity, or **over-refusal** (refusing innocuous prompts like *"how to kill a linux process"*).

---

### 2. Persona Vectors and Assistant Axis Activation Capping

Beyond weight-space regularization, recent mechanistic post-training work (Chen et al., 2025; Lu et al., 2026) controls behavioral drift directly in the residual stream $\mathbf{h}_\ell \in \mathbb{R}^d$ at layer $\ell$:

1. **Persona Vector Extraction:** Given responses $S^+$ exhibiting a trait (e.g., sycophancy) and $S^-$ suppressing it, the linear trait direction is the difference of mean residual activations:
   $$\mathbf{v}_\ell = \frac{1}{|S^+|}\sum_{i \in S^+} \mathbf{a}_\ell^{(i)} - \frac{1}{|S^-|}\sum_{j \in S^-} \mathbf{a}_\ell^{(j)}$$
2. **The Assistant Axis & Activation Capping (Lu et al., 2026):** Across $275+$ character archetypes, the first principal component of persona space aligns with the unit contrast vector $\mathbf{v}$ ($\|\mathbf{v}\|_2 = 1$) pointing toward the default analytical **Assistant** persona. In long, emotionally charged multi-turn conversations, the residual stream projection $p = \langle \mathbf{h}, \mathbf{v} \rangle$ can drift below a safe floor $\tau$, causing the model to abandon its assistant persona. **Activation Capping** applies a one-sided clamp along $\mathbf{v}$:
   $$\mathbf{h}' = \mathbf{h} - \mathbf{v} \cdot \min\bigl(\langle \mathbf{h}, \mathbf{v} \rangle - \tau,\; 0\bigr)$$

   Check both cases algebraically:
   - **Safe region ($p = \langle \mathbf{h}, \mathbf{v} \rangle \ge \tau$):** $\min(p - \tau, 0) = 0 \implies \mathbf{h}' = \mathbf{h}$ (zero intervention on normal queries).
   - **Drifted region ($p = \langle \mathbf{h}, \mathbf{v} \rangle < \tau$):** $\min(p - \tau, 0) = p - \tau < 0 \implies \mathbf{h}' = \mathbf{h} - (p - \tau)\mathbf{v}$. Take the inner product with $\mathbf{v}$ (using $\langle \mathbf{v}, \mathbf{v} \rangle = 1$):
     $$\langle \mathbf{h}', \mathbf{v} \rangle = \langle \mathbf{h}, \mathbf{v} \rangle - (p - \tau)\langle \mathbf{v}, \mathbf{v} \rangle = p - (p - \tau) = \tau$$
     The projection along $\mathbf{v}$ is restored to $\tau$, while every component orthogonal to $\mathbf{v}$ ($\langle \mathbf{u}, \mathbf{v} \rangle = 0$) is untouched.

---

### 3. Model Merging: Task Arithmetic, DARE, and TIES-Merging

Post-training recipes (`Tülu 3`, `OLMo 3`, `Command A`) routinely sweep multiple random seeds or train domain specialists $\theta_1, \dots, \theta_M$ starting from the same base checkpoint $\theta_{\text{base}}$ and merge their weights in parameter space without spending a single GPU-hour on extra training.

Define the **task vector** (weight delta) for model $m$ as:

$$\tau_m = \theta_m - \theta_{\text{base}}$$

1. **Linear Task Arithmetic (Ilharco et al., 2023):**
   $$\theta_{\text{merged}} = \theta_{\text{base}} + \sum_{m=1}^M \lambda_m \tau_m$$
   Why can naive summation degrade performance when $M$ is large? Because two models $\tau_1, \tau_2$ may modify the same parameter coordinate $i$ with **redundant** magnitudes or **opposing signs** ($\tau_{1,i} = +0.6,\; \tau_{2,i} = -0.6 \implies \text{cancels to } 0$).
2. **DARE — Drop And REscale (Yu et al., 2024):**
   Up to $90\%$–$99\%$ of fine-tuning delta coordinates $\tau_{m,i}$ can be randomly zeroed out provided the surviving coordinates are rescaled by $\frac{1}{1 - p}$ (exactly like inverted dropout!):
   $$m_i \sim \text{Bernoulli}(1 - p), \qquad \tilde{\tau}_{m,i} = \frac{m_i}{1 - p}\,\tau_{m,i}$$
   Take the expectation of $\tilde{\tau}_{m,i}$:
   $$\mathbb{E}[\tilde{\tau}_{m,i}] = \Pr[m_i = 1] \cdot \frac{\tau_{m,i}}{1 - p} + \Pr[m_i = 0] \cdot 0 = (1 - p)\frac{\tau_{m,i}}{1 - p} = \tau_{m,i}$$
   Because every linear layer computes $(\theta_{\text{base}} + \tilde{\tau}_m) x$, preserving the expectation of every coordinate preserves the expected layer activation while reducing overlap collisions between two merged models by $(1 - p)^2$ (at $p = 0.9$, collision rate drops by **$100\times$**, from $100\%$ to $1\%$!).
3. **TIES-Merging — Trim, Elect Sign, Disjoint Merge (Yadav et al., 2023):**
   - **Step 1 (Trim):** Zero out the smallest $(100 - k)\%$ magnitude coordinates in each $\tau_m$ to obtain $\hat{\tau}_m$.
   - **Step 2 (Elect Sign):** For each parameter coordinate $i$, sum the signed mass across all $M$ models to elect the dominant sign:
     $$\gamma_i = \text{sign}\!\left(\sum_{m=1}^M \hat{\tau}_{m,i}\right)$$
   - **Step 3 (Disjoint Merge):** Average *only* the models whose sign at coordinate $i$ agrees with $\gamma_i$:
     $$\tau_i^{\text{TIES}} = \frac{1}{|\mathcal{A}_i|} \sum_{m \in \mathcal{A}_i} \hat{\tau}_{m,i}, \qquad \mathcal{A}_i = \{\, m \mid \text{sign}(\hat{\tau}_{m,i}) = \gamma_i \,\}$$
   Check the demo's 3-model example on coordinate $0$ where $\hat{\tau}_{\cdot, 0} = [+0.6, +0.4, -0.2]$:
   - Naive average: $(0.6 + 0.4 - 0.2)/3 = 0.2667$ (diluted by model 3's opposing sign).
   - TIES: sum $= +0.8 > 0 \implies \gamma_0 = +1$; agreeing set $\mathcal{A}_0 = \{1, 2\} \implies (0.6 + 0.4)/2 = \mathbf{0.5000}$.

---

### 4. Post-Training Evaluation: `pass@k` vs. `pass^k`, Variance, and Decontamination

Evaluating reasoning and agentic models requires three mathematical safeguards:

1. **Unbiased `pass@k` (Chen et al., 2021) vs. Agentic Consistency `pass^k` (Yao et al., 2024 $\tau$-bench):**
   Suppose you generate $n \ge k$ independent samples on a coding or math problem and $c$ of them pass verification:
   - **`pass@k` (discovery / best-of-$k$):** Probability that a random subset of $k$ samples drawn without replacement contains **at least one** correct solution ($1$ minus the probability that all $k$ come from the $n - c$ wrong samples):
     $$\text{pass@}k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}} = 1 - \prod_{i=0}^{k-1} \frac{n - c - i}{n - i}$$
   - **`pass^k` (production agent reliability):** Probability that an autonomous tool-use agent succeeds on **every single one of $k$ independent customer trials**:
     $$\text{pass}^k = \left(\frac{c}{n}\right)^k$$
   For $n = 10, c = 8$ ($80\%$ single-try accuracy): `pass@5` is **$1.0000$ ($100\%$)**, whereas `pass^5` is **$0.80^5 = 0.3277$ ($32.8\%$)**!

??? question "Why is pass@k the wrong metric for production tool-use agents (and why pass^k is used instead)?"
    `pass@k` measures whether **at least one** of $k$ attempts succeeds, which assumes an oracle verifier at test time to discard the $k-1$ failures. Production agents modifying databases or files have no oracle undo button: a customer running the agent $k$ times needs **every** trial to succeed, which occurs with probability $\text{pass}^k = (c/n)^k$. See [Q&A: Why is pass@k the wrong metric for production tool-use agents?](qa.md#why-is-passk-the-wrong-metric-for-production-tool-use-agents-and-why-passk-is-used-instead).
2. **Evaluation Variance (`avg@K`):** On small reasoning benchmarks like AIME ($30$ questions), single-run temperature-$0.6$ sampling has standard deviation $\sigma_1 \approx 3.0$ points; reporting `avg@32` or `avg@64` shrinks standard error by $\sigma_1 / \sqrt{K}$ to $\approx 0.54$ points (`OLMo 3`).
3. **8-Gram Decontamination (`Tülu 3`):** Before SFT, DPO, or RLVR, remove any training prompt whose 8-gram token overlap (or 50-character substring match) hits held-out evaluation sets (`MATH-500`, `GPQA`, `HumanEval`, `TruthfulQA`).

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

---

**Part VII — and the full 48-lecture course — complete.**

You now hold both halves of the modern LLM stack in one mathematical model: how tokens are trained into weights (Parts VI–VII), how weights are served back into tokens (Parts I–V), and why modern RL post-training is where both halves fuse into a single loop.

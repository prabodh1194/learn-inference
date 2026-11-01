# Lecture 36b: Rejection Sampling, Best-of-N & Synthetic Data Engines

Before introducing online reinforcement learning (PPO in [Lecture 37](37-reward-models-and-ppo.md), DPO in [Lecture 38](38-dpo-and-offline-preference.md), GRPO in [Lecture 39](39-grpo-and-reinforcement-learning.md)), we must study the simplest and most widely deployed bridge between Supervised Fine-Tuning ([Lecture 36](36-sft-and-lora.md)) and reinforcement learning: **Rejection Sampling (RS)** and **Best-of-$N$ (BoN)**.

WebGPT, Anthropic's Helpful-and-Harmless Claude, Llama 2 Chat, DeepSeek-R1 Stage 3, and Qwen3 Stage 3 all rely on rejection sampling to filter synthetic completions using a reward model or verifier, then run standard SFT cross-entropy on the survivors.

---

### 1. Best-of-$N$ Sampling and Its Exact KL Divergence

In **Best-of-$N$ (BoN)** sampling at inference time, we sample $N$ independent completions $y_1, \dots, y_N \sim \pi_{\text{ref}}(\cdot \mid x)$, score each with a reward model $r(x, y_i)$, and return:

$$y_{\text{BoN}} = \arg\max_{1 \le i \le N} r(x, y_i)$$

BoN never updates model weights $\theta$, yet it induces a new sampling distribution $\pi_{\text{BoN}}(y \mid x)$ that is tilted toward high reward. How far does $\pi_{\text{BoN}}$ drift from $\pi_{\text{ref}}$ in Kullback-Leibler divergence $\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}})$?

#### Full Derivation of $\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}}) = \ln N - \frac{N-1}{N}$

Assume $r(x, y)$ has a continuous cumulative distribution function $F(r)$ under $y \sim \pi_{\text{ref}}(\cdot \mid x)$ (no ties). By the probability integral transform, the quantile $u = F(r(x, y)) \in [0, 1]$ is uniformly distributed under the reference policy:

$$q(u) = 1, \qquad u \in [0, 1]$$

Drawing $N$ independent completions from $\pi_{\text{ref}}$ and picking the maximum reward is identical to drawing $N$ independent uniform random variables $u_1, \dots, u_N \sim \text{Uniform}(0, 1)$ and selecting $u_{\max} = \max_{1 \le i \le N} u_i$:

1. **Cumulative distribution function of $u_{\max}$:** The maximum of $N$ independent draws is $\le u$ if and only if every single draw $u_i \le u$:
   $$F_{\text{BoN}}(u) = \Pr[\max_{1 \le i \le N} u_i \le u] = \prod_{i=1}^N \Pr[u_i \le u] = u^N$$
2. **Probability density function $p_{\text{BoN}}(u)$:** Differentiate $F_{\text{BoN}}(u)$ with respect to $u$:
   $$p_{\text{BoN}}(u) = \frac{d}{du}\left(u^N\right) = N u^{N-1}$$
3. **Set up the KL divergence integral:** Because a monotonic bijective mapping preserves KL divergence,
   $$\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}}) = \int_0^1 p_{\text{BoN}}(u) \ln \frac{p_{\text{BoN}}(u)}{q(u)} \, du = \int_0^1 N u^{N-1} \ln\left(N u^{N-1}\right) du$$
4. **Split the logarithm $\ln(N u^{N-1}) = \ln N + (N-1)\ln u$:**
   $$\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}}) = \ln N \int_0^1 N u^{N-1} \, du + N(N-1) \int_0^1 u^{N-1} \ln u \, du$$
5. **Evaluate both integrals explicitly:**
   - First integral:
     $$\int_0^1 N u^{N-1} \, du = \left[u^N\right]_0^1 = 1 - 0 = 1 \implies \ln N \cdot 1 = \ln N$$
   - Second integral via integration by parts ($\text{let } v = \ln u \implies dv = \frac{1}{u}du$, $\text{and } dw = u^{N-1}du \implies w = \frac{u^N}{N}$):
     $$\int_0^1 u^{N-1} \ln u \, du = \underbrace{\left[\frac{u^N}{N} \ln u\right]_0^1}_{0 - 0 = 0} - \int_0^1 \frac{u^N}{N} \cdot \frac{1}{u} \, du = -\frac{1}{N} \int_0^1 u^{N-1} \, du = -\frac{1}{N}\left[\frac{u^N}{N}\right]_0^1 = -\frac{1}{N^2}$$
   - Multiply by $N(N-1)$:
     $$N(N-1)\left(-\frac{1}{N^2}\right) = -\frac{N-1}{N}$$
6. **Combine both terms:**
   $$\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}}) = \ln N - \frac{N-1}{N}$$

At $N = 16$, $\mathbb{D}_{\text{KL}} = \ln(16) - \frac{15}{16} = 2.7726 - 0.9375 = 1.8351\text{ nats}$, while paying a $16\times$ inference compute multiplier.

---

### 2. Rejection Sampling Fine-Tuning (RS / RAFT / STaR)

**Rejection Sampling** amortizes Best-of-$N$'s inference cost into the model weights: generate $N$ completions for each of $M$ training prompts, score the $M \times N$ reward matrix $R \in \mathbb{R}^{M \times N}$, select the winners, and run SFT on the winners.

Given the $5 \times 4$ reward matrix ($M = 5$ prompts, $N = 4$ completions per prompt):

$$R = \begin{bmatrix}
\mathbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \mathbf{0.8} & 0.6 & 0.5 \\
\mathbf{0.9} & 0.3 & 0.4 & \underline{0.7} \\
0.2 & 0.5 & \mathbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & \mathbf{0.6}
\end{bmatrix}$$

There are two selection rules, and their difference is critical for training diversity:

1. **Top-per-prompt (`argmax` per row, bolded):** Selects $j^*(i) = \arg\max_j R_{i,j}$ for every prompt $i \in \{0, \dots, 4\}$, yielding indices `[(0,0,0.7), (1,1,0.8), (2,0,0.9), (3,2,0.8), (4,3,0.6)]` with mean reward $0.76$. Every prompt is represented exactly once.
2. **Top-$K$ overall (global sort across $R_{\text{flat}}$):** Flattens $R$ into $M \cdot N = 20$ entries and takes the top $K = 5$, yielding `[(2,0,0.9), (1,1,0.8), (3,2,0.8), (0,0,0.7), (2,3,0.7)]` with higher mean reward $0.78$ — **but prompt 2 is sampled twice and hard prompt 4 is dropped completely!** In reasoning and math pipelines, dropping hard prompts starves the model on the exact problems it needs to learn.

??? question "Why does global top-K rejection sampling hurt reasoning compared to per-prompt selection?"
    In a global top-$K$ sort over the $M \times N$ reward matrix, easy prompts where the model already scores $0.9+$ contribute multiple completions, while hard reasoning prompts where only one rollout scored $0.6$ get dropped completely. Per-prompt `argmax` selection preserves uniform coverage across easy and hard prompts alike. See [Q&A: Why does global top-K rejection sampling hurt reasoning compared to per-prompt selection?](qa.md#why-does-global-top-k-rejection-sampling-hurt-reasoning-compared-to-per-prompt-selection).

---

### 3. Synthetic Data Engines, Persona Synthesis & Constitutional AI

Modern SFT and preference datasets (`Tülu 3`, `OpenThoughts 3`, `UltraFeedback`, `Qwen3`) are $500\text{M}$ to $10\text{B}$ tokens of **synthetic data**. How do labs prevent synthetic training from collapsing onto narrow, repetitive phrasing (**model collapse**, Shumailov et al. 2024)?

1. **Persona-Driven Prompt Synthesis (Ge et al. 2024 / Tülu 3):** Asking an LLM *"Generate 10,000 math or coding questions"* yields highly repetitive prompts. Conditioning each seed generation on one of $200{,}000+$ diverse **personas** extracted from web text (e.g., *"a marine acoustic engineer modeling sonar refraction"*) forces broad coverage of the input prompt space $\mathcal{X}$.
2. **Multi-Teacher Generation & Verification:** Responses are sampled from a diverse pool of frontier teachers (or reasoning checkpoints like `QwQ-32B` / `DeepSeek-R1`) and filtered by deterministic verifiers (unit tests, symbolic math checkers) or checklist-guided judges.
3. **Constitutional AI / RLAIF (Bai et al., 2022):** Given an initial response $y_0 \sim \pi_{\text{SFT}}(\cdot \mid x)$, the model critiques $y_0$ against a written constitution principle $c_k$ and generates a revised response $y_1 \sim \pi(\cdot \mid x, y_0, \text{critique}, c_k)$ for SFT, then uses AI feedback pairs $(y_a, y_b)$ judged against constitutional principles to train the preference model.

```bash
uv run python book/code/rejection_sampling.py
```

```text
Best-of-N Analytical KL Budget: KL = ln(N) - (N - 1)/N
----------------------------------------------------------------
  N =  1 | KL(BoN || ref) = 0.0000 nats | E[max Z] ~ 0.000 sigma
  N =  2 | KL(BoN || ref) = 0.1931 nats | E[max Z] ~ 1.177 sigma
  N =  4 | KL(BoN || ref) = 0.6363 nats | E[max Z] ~ 1.665 sigma
  N =  8 | KL(BoN || ref) = 1.2044 nats | E[max Z] ~ 2.039 sigma
  N = 16 | KL(BoN || ref) = 1.8351 nats | E[max Z] ~ 2.355 sigma
  N = 32 | KL(BoN || ref) = 2.4970 nats | E[max Z] ~ 2.633 sigma
  N = 64 | KL(BoN || ref) = 3.1745 nats | E[max Z] ~ 2.884 sigma

Rejection Sampling Selection (5 prompts x 4 completions):
  top_per_prompt : [(0, 0, 0.7), (1, 1, 0.8), (2, 0, 0.9), (3, 2, 0.8), (4, 3, 0.6)] (mean R = 0.76)
  top_5_overall  : [(2, 0, 0.9), (1, 1, 0.8), (3, 2, 0.8), (0, 0, 0.7), (2, 3, 0.7)] (mean R = 0.78)
```

---

*Next: [Lecture 37: Reward Modeling & Proximal Policy Optimization (PPO)](37-reward-models-and-ppo.md)*

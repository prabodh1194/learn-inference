# 36b. Rejection sampling, Best-of-N, and synthetic data

**Demo:** `book/code/rejection_sampling.py`
**Moves:** derives the exact $\ln N - \frac{N-1}{N}$ KL cost of Best-of-$N$ and prevents easy prompts from crowding out hard ones during rejection sampling
**Prereq:** [06b. Search-based decoding](06b-search-based-decoding.md), [36. Supervised fine-tuning (SFT), LoRA, and QLoRA](36-sft-and-lora.md)

---

## The problem

Supervised fine-tuning (Lecture 36) imitates whatever responses sit in your
training set. What if you do not have human-written solutions for a million hard
math and coding prompts, or what if your SFT model already solves a problem 1
time out of 8 and you want it to succeed 8 times out of 8?

Before building online reinforcement learning (PPO in Lecture 37, DPO in Lecture
38, GRPO in Lecture 39), there is a much simpler bridge that needs no policy
gradient at all: sample $N$ candidate answers per prompt from your current
model, score them with a verifier or reward model, keep the winners, and run
standard SFT cross-entropy on those winners.

WebGPT, Claude, Llama 2, DeepSeek-R1 Stage 3, and Qwen3 Stage 3 all rely on this
**Rejection Sampling (RS)** loop. To use it well, we need to answer three
questions. How far does picking the best of $N$ samples move the output
distribution in KL divergence? When filtering an $M \times N$ table of candidate
answers, why does taking the top $K$ answers overall hurt reasoning compared to
taking the best answer per prompt? And how do synthetic data pipelines generate
millions of prompts without collapsing onto repetitive phrasing?

---

## See it

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

Look at both parts of the output. In the top table, drawing $N = 16$ samples and
picking the highest scorer shifts the policy by **1.8351 nats** of KL divergence
at a cost of $16\times$ inference compute. In the bottom table, global top-5
selection gets a slightly higher mean reward ($0.78$ vs. $0.76$), but it samples
prompt 2 twice (`(2, 0)` and `(2, 3)`) and drops hard prompt 4 altogether.

---

## The idea

### 1. Best-of-$N$ sampling and its exact KL divergence

In **Best-of-$N$ (BoN)** sampling at test time, we draw $N$ independent
completions $y_1, \dots, y_N \sim \pi_{\text{ref}}(\cdot \mid x)$, score each one
with a reward model $r(x, y_i)$, and return the highest-scoring completion.

$$y_{\text{BoN}} = \arg\max_{1 \le i \le N} r(x, y_i)$$

Even though BoN never changes a single weight in $\theta$, picking the maximum of
$N$ draws creates a new effective distribution $\pi_{\text{BoN}}(y \mid x)$ that
tilts toward high reward. How far does $\pi_{\text{BoN}}$ drift from the base
distribution $\pi_{\text{ref}}$ in KL divergence
$\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}})$?

We can derive the exact formula by mapping rewards to uniform quantiles. Assume
the reward $r(x, y)$ has a continuous cumulative distribution function $F(r)$
when $y$ is drawn from $\pi_{\text{ref}}(\cdot \mid x)$, so there are no ties.
For any completion $y$, define its reward quantile $u = F(r(x, y)) \in [0, 1]$.
By the probability integral transform, $u$ is uniformly distributed on $[0, 1]$
under the reference policy, with density $q(u) = 1$.

Drawing $N$ independent completions from $\pi_{\text{ref}}$ and picking the
highest reward is therefore the exact same random process as drawing $N$
independent uniform numbers $u_1, \dots, u_N \sim \text{Uniform}(0, 1)$ and
keeping their maximum, $u_{\max} = \max_{1 \le i \le N} u_i$.

First, find the cumulative distribution function $F_{\text{BoN}}(u)$ of
$u_{\max}$. The maximum of $N$ independent draws is at most $u$ if and only if
every single draw $u_i$ is at most $u$.

$$F_{\text{BoN}}(u) = \Pr[\max_{1 \le i \le N} u_i \le u] = \prod_{i=1}^N \Pr[u_i \le u] = u^N$$

Second, differentiate $F_{\text{BoN}}(u)$ with respect to $u$ to get the
probability density function $p_{\text{BoN}}(u)$ of the winning quantile.

$$p_{\text{BoN}}(u) = \frac{d}{du}\left(u^N\right) = N u^{N-1}$$

Third, write the KL divergence between $p_{\text{BoN}}(u)$ and the uniform base
density $q(u) = 1$. Because a strictly increasing map preserves KL divergence,
the KL in quantile space equals the KL in completion space.

$$\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}}) = \int_0^1 p_{\text{BoN}}(u) \ln \frac{p_{\text{BoN}}(u)}{q(u)} \, du = \int_0^1 N u^{N-1} \ln\left(N u^{N-1}\right) du$$

Fourth, expand the logarithm using $\ln(N u^{N-1}) = \ln N + (N-1)\ln u$ to
split the integral into two terms.

$$\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}}) = \ln N \int_0^1 N u^{N-1} \, du + N(N-1) \int_0^1 u^{N-1} \ln u \, du$$

Fifth, evaluate each integral in turn. The first integral is the total
probability mass of $p_{\text{BoN}}(u)$ on $[0, 1]$, which equals $1$:

$$\int_0^1 N u^{N-1} \, du = \left[u^N\right]_0^1 = 1 - 0 = 1 \implies \ln N \cdot 1 = \ln N$$

For the second integral, integrate by parts with $v = \ln u$ (so
$dv = \frac{1}{u}du$) and $dw = u^{N-1}du$ (so $w = \frac{u^N}{N}$):

$$\int_0^1 u^{N-1} \ln u \, du = \underbrace{\left[\frac{u^N}{N} \ln u\right]_0^1}_{0 - 0 = 0} - \int_0^1 \frac{u^N}{N} \cdot \frac{1}{u} \, du = -\frac{1}{N} \int_0^1 u^{N-1} \, du = -\frac{1}{N}\left[\frac{u^N}{N}\right]_0^1 = -\frac{1}{N^2}$$

Multiply that result by the front factor $N(N-1)$:

$$N(N-1)\left(-\frac{1}{N^2}\right) = -\frac{N-1}{N}$$

Finally, add the two evaluated terms together to get the closed-form KL
divergence of Best-of-$N$.

$$\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}}) = \ln N - \frac{N-1}{N}$$

Check the formula at $N = 1$: $\ln(1) - 0 = 0$ nats. At $N = 16$, we get
$\ln(16) - \frac{15}{16} = 2.7726 - 0.9375 = 1.8351\text{ nats}$. Because KL
grows only logarithmically with $N$ while inference cost grows linearly with
$N$, Best-of-$N$ quickly becomes an expensive way to buy large policy shifts at
serving time.

---

### 2. Rejection sampling fine-tuning: per-prompt vs. global selection

**Rejection Sampling (RS)**, also called RAFT or STaR, moves that $N\times$
sampling bill from serving time to training time. We sample $N$ completions for
each of $M$ training prompts, score all $M \times N$ candidates to form a reward
matrix $R \in \mathbb{R}^{M \times N}$, keep the winning completions, and fine-tune
the model on those winners with standard SFT loss.

Look at the $5 \times 4$ reward matrix from the demo ($M = 5$ prompts, $N = 4$
completions per prompt).

$$R = \begin{bmatrix}
\mathbf{0.7} & 0.3 & 0.5 & 0.2 \\
0.4 & \mathbf{0.8} & 0.6 & 0.5 \\
\mathbf{0.9} & 0.3 & 0.4 & \underline{0.7} \\
0.2 & 0.5 & \mathbf{0.8} & 0.6 \\
0.5 & 0.4 & 0.3 & \mathbf{0.6}
\end{bmatrix}$$

Once you have the matrix $R$, there are two ways to pick $5$ training examples
from it.

Under **per-prompt selection** (`argmax` per row, shown in bold), we pick the
best completion $j^*(i) = \arg\max_j R_{i,j}$ inside each row
$i \in \{0, \dots, 4\}$. That gives `[(0,0,0.7), (1,1,0.8), (2,0,0.9), (3,2,0.8), (4,3,0.6)]`,
with a mean reward of $0.76$. Every prompt in your dataset appears in the SFT
mix once.

Under **global top-$K$ selection**, we flatten $R$ into $M \cdot N = 20$ numbers
and take the top $K = 5$ entries overall. That gives
`[(2,0,0.9), (1,1,0.8), (3,2,0.8), (0,0,0.7), (2,3,0.7)]`, with a higher mean
reward of $0.78$. Look at what global sorting did: it picked two completions from
easy prompt 2 (`0.9` and `0.7`) and **threw away hard prompt 4 altogether**
because its best rollout scored `0.6`.

??? question "Why does global top-K rejection sampling hurt reasoning compared to per-prompt selection?"
    When you sort all $M \times N$ rollouts globally, easy prompts where the
    model already scores $0.9+$ contribute several completions each, while hard
    math or coding prompts where only one rollout reached $0.6$ get dropped from
    the training set. Dropping the hardest prompts starves the model on the exact
    problems it has not yet mastered. Per-prompt selection guarantees every
    prompt stays in the training mix.
    [Full answer](qa.md#why-does-global-top-k-rejection-sampling-hurt-reasoning-compared-to-per-prompt-selection)

---

### 3. Synthetic data engines, persona synthesis, and Constitutional AI

Where do the millions of prompts and teacher responses in modern post-training
mixes (`Tülu 3`, `OpenThoughts 3`, `UltraFeedback`, `Qwen3`) come from? Almost
all of them are generated by models. Yet if you simply ask an LLM to *"write
10,000 math questions,"* it repeats the same few templates over and over.
Training on narrow synthetic outputs flattens the tail of the distribution, a
failure mode called **model collapse** (Shumailov et al., 2024).

Modern synthetic data pipelines avoid collapse by combining three mechanisms.
First, to keep prompts diverse, **persona-driven synthesis** (Ge et al., 2024;
Tülu 3) mines $200{,}000+$ specific personas from web text (such as *"a marine
acoustic engineer modeling sonar refraction"*) and conditions each prompt
generation call on a different persona. That forces the generated questions to
span the full input domain.

Second, to keep answers accurate, **multi-teacher generation and verification**
samples candidate answers from a pool of strong models (such as `DeepSeek-R1` or
`QwQ-32B`) at high temperature and filters them through deterministic verifiers
(unit tests, symbolic math checkers) or rubric-based judges before any completion
enters the SFT set.

Third, for safety and helpfulness where no unit test exists, **Constitutional AI
and RLAIF** (Bai et al., 2022) use a written list of principles. Starting from an
initial answer $y_0 \sim \pi_{\text{SFT}}(\cdot \mid x)$, the model critiques
$y_0$ against a principle $c_k$ and writes a revised answer
$y_1 \sim \pi(\cdot \mid x, y_0, \text{critique}, c_k)$ for SFT. It then judges
pairs of candidate answers $(y_a, y_b)$ against those same principles to build
preference pairs for Lectures 37 and 38 without waiting on human labelers.

---

## What you should see

Running `book/code/rejection_sampling.py` shows that Best-of-$N$ follows the
exact analytical curve $\mathbb{D}_{\text{KL}} = \ln N - \frac{N-1}{N}$, paying
linear inference compute for logarithmic KL gains. When you distill those gains
back into the weights with Rejection Sampling, per-prompt selection keeps every
hard prompt in the training mix instead of letting easy prompts crowd them out.

---

## Go deeper

- [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) (Bai et al., 2022), Appendix D derives $\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}}) = \ln N - \frac{N-1}{N}$.
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) (Bai et al., 2022), self-critique revision and RLAIF preference labeling.
- [Scaling Synthetic Data Creation with 1,000,000,000 Personas](https://arxiv.org/abs/2406.20094) (Ge et al., 2024), persona-conditioned prompt synthesis.
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) (Lambert et al., 2024), open-source synthetic SFT, preference, and RLVR curation recipes.

---

## Check yourself

1. In the derivation of $\mathbb{D}_{\text{KL}}(\pi_{\text{BoN}} \,\|\, \pi_{\text{ref}})$,
   why is the cumulative distribution function of the winning quantile equal to
   $u^N$, and how does integration by parts produce the $-\frac{N-1}{N}$ term?
2. How many completions $N$ do you need for Best-of-$N$ to reach a KL divergence
   of roughly $1.20$ nats from $\pi_{\text{ref}}$?
3. When does global top-$K$ rejection sampling outperform per-prompt selection,
   and when does it hurt?

---

## Next

**[37. Reward modeling and Proximal Policy Optimization (PPO)](37-reward-models-and-ppo.md)**:
Rejection sampling only filters whole completions the model already knows how to
generate. Now train a Bradley-Terry reward model and optimize the policy directly
with PPO, and count why its 4-model loop demands $36\Psi$ bytes of VRAM.

```bash
uv run python book/code/rlhf_memory.py
```

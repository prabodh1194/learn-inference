# 30b. Scaling laws, schedules, and mid-training

**Demo:** `book/code/scaling_laws.py`
**Moves:** picks model size $N$, token budget $D$, batch size $B$, and context extension schedule before a cluster run starts
**Prereq:** [30. The training ledger](30-training-arithmetic.md)

---

## The problem

Lecture 30 showed that one training token through a dense model of $N$
parameters costs $6N$ FLOPs, while one inference token costs $2N$ FLOPs. That
$3:1$ ratio per token hides a much larger split over a model's life. You train a
model once on $D_{\text{train}}$ tokens, and then you serve it to users across
$D_{\text{infer}}$ tokens for months or years.

Before you launch a pretraining run, you have to answer four questions. How do
you split a fixed compute budget between parameters $N$ and training tokens $D$?
Why do modern small models train on thousands of times more tokens than classic
scaling laws recommend? How big can you make the global batch size $B$ before
extra GPUs stop helping? And how do you anneal the learning rate and stretch the
context window from $4{,}096$ to $32{,}768$ tokens during **mid-training** (the
bridge stage between raw web pretraining and post-training)?

---

## See it

```bash
uv run python book/code/scaling_laws.py
```

Pure arithmetic, no GPU required:

```text
Over-training vs. Chinchilla (100T lifetime served tokens)
------------------------------------------------------------------------
Chinchilla 2.8B (20 tok/param)       | L=2.341 | train=   0.9 ZFLOP | infer= 560.0 ZFLOP | total= 560.9 ZFLOP
Over-trained 1.4B (300 tok/param)    | L=2.234 | train=   3.5 ZFLOP | infer= 280.0 ZFLOP | total= 283.5 ZFLOP
Qwen3-0.6B scale (60,000 tok/param)  | L=2.177 | train= 129.6 ZFLOP | infer= 120.0 ZFLOP | total= 249.6 ZFLOP

YaRN attention temperature scale sqrt(1/t) = 0.1 * ln(s) + 1:
  s =  1x -> sqrt(1/t) = 1.0000, t = 1.0000
  s =  4x -> sqrt(1/t) = 1.1386, t = 0.7713
  s =  8x -> sqrt(1/t) = 1.2079, t = 0.6853
  s = 32x -> sqrt(1/t) = 1.3466, t = 0.5515
```

Look at the lifetime column. Training a $1.4\text{B}$ model for $300$ tokens per
parameter costs nearly $4\times$ as many training FLOPs as a Chinchilla-optimal
$2.8\text{B}$ model, yet it reaches a *lower* validation loss ($2.234$ vs.
$2.341$) and cuts total lifetime compute across $100\text{T}$ served tokens in
half ($283.5$ vs. $560.9\text{ ZFLOP}$).

---

## The idea

### 1. Chinchilla compute-optimal scaling

Suppose you have a fixed pretraining compute budget $C = 6ND$ FLOPs and you do
not care what inference will cost later. Making the model larger ($N$) lets it
represent richer patterns, but a fixed budget $C$ then forces you to train on
fewer tokens ($D = C / 6N$). Training on more tokens ($D$) reduces sample noise,
but forces a smaller model ($N$).

Hoffmann et al. (2022) measured how validation cross-entropy loss $L(N, D)$
falls as both $N$ and $D$ grow. They modeled the loss as the irreducible entropy
$E$ of natural text plus two power-law error terms.

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Fitting this curve to hundreds of training runs gave $E = 1.69$, $A = 406.4$,
$B = 410.7$, $\alpha = 0.34$, and $\beta = 0.28$. We want the pair $N^*, D^*$
that minimizes $L(N, D)$ subject to the budget constraint $6ND = C$.

First, solve the constraint $6ND = C$ for $D$. That gives $D = C / (6N)$. Now
substitute $D = C / (6N)$ into $L(N, D)$ so the loss depends on $N$ alone.

$$L(N) = E + A N^{-\alpha} + B \left(\frac{C}{6N}\right)^{-\beta} = E + A N^{-\alpha} + B \left(\frac{C}{6}\right)^{-\beta} N^{\beta}$$

Next, differentiate $L(N)$ with respect to $N$. At the minimum, that derivative
equals zero.

$$\frac{dL}{dN} = -\alpha A N^{-\alpha - 1} + \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\beta - 1} = 0$$

Move the negative term to the right-hand side and multiply both sides by
$N^{\alpha + 1}$. That collects both powers of $N$ on the left.

$$\beta B \left(\frac{C}{6}\right)^{-\beta} N^{\alpha + \beta} = \alpha A$$

Divide both sides by $\beta B (C/6)^{-\beta}$ to isolate $N^{\alpha + \beta}$.

$$N^{\alpha + \beta} = \frac{\alpha A}{\beta B} \left(\frac{C}{6}\right)^{\beta}$$

Finally, raise both sides to the power $\frac{1}{\alpha + \beta}$. That gives
the compute-optimal parameter count $N^*(C)$.

$$N^*(C) = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}} \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha + \beta}}$$

By symmetry, plugging $N^*(C)$ back into $D^*(C) = \frac{C}{6 N^*(C)}$ gives the
compute-optimal token count.

$$D^*(C) = \left(\frac{\beta B}{\alpha A}\right)^{\frac{1}{\alpha + \beta}} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha + \beta}}$$

Plug in the fitted exponents $\alpha = 0.34$ and $\beta = 0.28$. Their sum is
$\alpha + \beta = 0.62$.

$$\frac{\beta}{\alpha + \beta} = \frac{0.28}{0.62} \approx 0.452, \qquad \frac{\alpha}{\alpha + \beta} = \frac{0.34}{0.62} \approx 0.548$$

Because $\alpha$ and $\beta$ are nearly equal, $N^*$ and $D^*$ both grow roughly
as $C^{0.5}$. Dividing $D^*$ by $N^*$ across typical budgets gives the
**Chinchilla rule of thumb**: train for about **20 tokens per parameter**, or
$D^* \approx 20 N^*$.

For `Qwen3-0.6B` with $N = 6 \times 10^8$, Chinchilla recommends stopping at
$D^* = 20 \times 6 \times 10^8 = 12\text{B}$ tokens. Instead, the Qwen3 team
trained `Qwen3-0.6B` on **$36\text{ trillion}$ tokens**, which is $3{,}000\times$
the Chinchilla ratio, or $60{,}000$ tokens per parameter.

---

### 2. Inference-aware scaling: why modern models over-train

Chinchilla minimized $L(N, D_{\text{train}})$ under a fixed *training* budget
$C_{\text{train}} = 6 N D_{\text{train}}$ and treated inference as free. In
production, you pay the training bill once. Then you pay the serving bill on
every generated token across $D_{\text{infer}}$ lifetime tokens.

$$C_{\text{total}}(N, D_{\text{train}}) = \underbrace{6 N D_{\text{train}}}_{\text{paid once}} + \underbrace{2 N D_{\text{infer}}}_{\text{paid across all users}}$$

Suppose your fleet will serve $D_{\text{infer}} = 100\text{T}$ lifetime tokens.
Compare three ways to build the model.

Start with a Chinchilla-scaled $2.8\text{B}$ model trained for $20$ tokens per
parameter, so $N = 2.8 \times 10^9$ and $D_{\text{train}} = 5.6 \times 10^{10}$.
Its loss reaches $L = 1.69 + 0.250 + 0.401 = 2.341$. The training run costs
$0.94\text{ ZFLOP}$, while serving $100\text{T}$ tokens costs
$560.0\text{ ZFLOP}$. Total lifetime compute comes to **$560.9\text{ ZFLOP}$**.

Now cut the model in half to $1.4\text{B}$ parameters and over-train it for
$300$ tokens per parameter, with $D_{\text{train}} = 4.2 \times 10^{11}$. Its
loss drops to $L = 1.69 + 0.316 + 0.228 = 2.234$, which beats the $2.8\text{B}$
Chinchilla model. Training costs more at $3.53\text{ ZFLOP}$, but serving
$100\text{T}$ tokens now costs only $280.0\text{ ZFLOP}$. Total lifetime compute
falls to **$283.5\text{ ZFLOP}$**, nearly $2\times$ cheaper overall, and every
decode step moves half as many weight bytes from HBM.

Push that trade-off to `Qwen3-0.6B` at $60{,}000$ tokens per parameter, where
$N = 0.6 \times 10^9$ and $D_{\text{train}} = 3.6 \times 10^{13}$. The power-law
curve gives $L = 1.69 + 0.421 + 0.066 = 2.177$. Training takes
$129.6\text{ ZFLOP}$ and serving takes $120.0\text{ ZFLOP}$, for
**$249.6\text{ ZFLOP}$** lifetime. Better still, the finished model fits in
$1.2\text{ GB}$ of BF16 weights on a phone or laptop.

??? question "Why did Chinchilla say 20 tokens per parameter while Llama 3 and Qwen3 train on 1,000 to 60,000 tokens per parameter?"
    Chinchilla answered a narrow question: given a fixed pretraining budget
    $C_{\text{train}}$, what $(N, D)$ gives the lowest loss at the end of
    training? It priced inference at zero. In a real deployment, inference FLOPs
    ($2 N D_{\text{infer}}$) and HBM weight traffic ($2N$ bytes per decoded
    token, Lecture 02) dwarf the one-time training run. Spending $4\times$ to
    $100\times$ more FLOPs once during pretraining to shrink $N$ by $2\times$ to
    $4\times$ pays back on every token the model ever generates.
    [Full answer](qa.md#why-did-chinchilla-say-20-tokens-per-parameter-while-llama-3-and-qwen3-train-on-1000-tokens-per-parameter)

---

### 3. Critical batch size and the batch-size ramp

If `Qwen3-0.6B` needs $36\text{T}$ training tokens, why not finish the run in an
afternoon by spreading one giant batch $B$ across a million GPUs?

Each sample $x_i$ gives a noisy gradient $g_i = \nabla_\theta \ell(x_i)$ around
the true population gradient $G = \mathbb{E}[g_i]$ with covariance
$\Sigma = \text{Cov}(g_i)$. Averaging $B$ independent samples produces a batch
gradient $\bar{g}_B$ with mean $G$ and covariance $\frac{1}{B}\Sigma$. When $B$
is small, the noise term $\frac{1}{B}\text{tr}(H\Sigma)$ is large, so doubling
$B$ cuts the noise in half and lets you take twice as large an effective step.
Once $B$ is large enough that the noise $\frac{1}{B}\text{tr}(H\Sigma)$ is
already smaller than the true signal $\|G\|_H^2$, averaging even more samples in
the same step barely improves the update direction.

McCandlish et al. (2018) showed that this saturation point sits at the
**gradient noise scale**, also called the **critical batch size**.

$$B_{\text{crit}} \approx B_{\text{noise}} = \frac{\text{tr}(\Sigma)}{\|G\|^2}$$

Let $S_{\min}$ be the minimum number of serial optimizer steps needed at infinite
batch size, and let $E_{\min} = B_{\text{crit}} S_{\min}$ be the minimum number
of training tokens needed in the small-batch limit. At any batch size $B$, the
required serial steps $S(B)$ and total data tokens $E(B) = B \cdot S(B)$ follow
a simple trade-off.

$$\frac{S(B)}{S_{\min}} = 1 + \frac{B_{\text{crit}}}{B}, \qquad \frac{E(B)}{E_{\min}} = 1 + \frac{B}{B_{\text{crit}}}$$

At $B = B_{\text{crit}}$, you pay $2 S_{\min}$ serial steps and $2 E_{\min}$
tokens. Below $B_{\text{crit}}$, doubling $B$ nearly halves wall-clock steps
$S(B)$ with almost no extra token waste. In that regime, you scale the learning
rate as $\eta \propto B$ for SGD or $\eta \propto \sqrt{B}$ for Adam. Above
$B_{\text{crit}}$, the batch gradient is already clean, so doubling $B$ barely
moves $S(B)$ while doubling the tokens burned per step.

Notice how $B_{\text{crit}} = \text{tr}(\Sigma)/\|G\|^2$ changes during a run.
In the first few billion tokens, the model is far from a minimum, so the true
gradient norm $\|G\|$ is large and $B_{\text{crit}}$ is small. As loss drops,
$\|G\|$ shrinks while the per-sample noise $\text{tr}(\Sigma)$ stays roughly
flat, so $B_{\text{crit}}$ grows by $10\times$ to $20\times$. That is why
DeepSeek-V3 and Llama 3 **ramp the global batch size** during pretraining,
starting around $4\text{M}$ tokens per step and stepping up to $16\text{M}$ or
$60\text{M}$ tokens once $B_{\text{crit}}$ has grown large enough to absorb the
wider parallelism.

---

### 4. Warmup-Stable-Decay (WSD) schedules and $\mu\text{P}$

Classic pretraining uses a **Cosine Decay** learning-rate schedule after $T_w$
warmup steps over a fixed horizon $T_{\max}$.

$$\eta_{\text{cos}}(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \frac{t - T_w}{T_{\max} - T_w}\right)\right)$$

Cosine decay forces a painful commitment: you have to pick the final step count
$T_{\max}$ before step 1. You cannot stop early at $0.5\,T_{\max}$ because the
learning rate is still high and the weights are bouncing around the walls of the
loss valley. You also cannot keep training smoothly past $T_{\max}$ once the
learning rate has hit $\eta_{\min}$.

Modern pretraining runs such as MiniCPM, DeepSeek-V3, OLMo 2, and Llama 3
replace Cosine with **Warmup-Stable-Decay**, or **WSD**, also called a
trapezoidal schedule.

$$\eta_{\text{WSD}}(t) = \begin{cases}
\eta_{\max} \frac{t}{T_w} & 0 \le t \le T_w \\[4pt]
\eta_{\max} & T_w < t \le T_s \\[4pt]
\eta_{\max} \left(1 - \frac{t - T_s}{T_{\max} - T_s}\right) & T_s < t \le T_{\max}
\end{cases}$$

WSD separates bulk learning from final convergence. During the **Stable phase**
from $T_w$ to $T_s$, the trainer holds $\eta = \eta_{\max}$ for $80\%$ to $90\%$
of the run without needing a predetermined stopping step $T_{\max}$. Whenever
you want a deployable model, whether at $1\text{T}$, $10\text{T}$, or
$36\text{T}$ tokens, you branch off a checkpoint and run the rapid **Decay
phase** over the last $10\%$ to $20\%$ of tokens while keeping the main stable
trunk running. Better still, during that decay branch you swap the training mix
from noisy web crawl to curated math, code, and reasoning data, a stage now
called **mid-training** or **annealing**.

How do you pick the peak learning rate $\eta_{\max}$ for a $70\text{B}$ model
when a single trial run costs millions of dollars? Standard initialization breaks
when you widen a network: as hidden width $d_{\text{model}}$ grows, activation
updates $\Delta h_\ell$ blow up unless you re-tune $\eta_{\max}$ at every scale.
**Maximal Update Parametrization**, or $\mu\text{P}$, from Yang et al. (2022)
fixes the scaling law of the parameterization itself. Under $\mu\text{P}$,
hidden weights initialize with variance $1 / d_{\text{model}}$ and scale their
Adam learning rate inversely with width.

$$\eta_{\text{hidden}} = \eta_{\text{base}} \cdot \frac{d_{\text{proxy}}}{d_{\text{target}}}$$

Meanwhile, input embeddings keep $\eta_{\text{embed}} = \eta_{\text{base}}$ and
the output readout scales by $d_{\text{proxy}} / d_{\text{target}}$. With those
factors in place, every layer's normalized feature update
$\|\Delta h_\ell\|_2 / \sqrt{d_{\text{model}}}$ stays $\Theta(1)$ at any width,
so the optimal $\eta_{\text{base}}$ found on a cheap $40\text{M}$ proxy model
transfers directly to a $70\text{B}$ run.

---

### 5. Mid-training long-context extension: RoPE scaling and YaRN

Training at sequence length $s = 32{,}768$ or $131{,}072$ for all $36\text{T}$
pretraining tokens would multiply quadratic attention FLOPs by $8\times$ to
$32\times$. Instead, models like `Qwen3` run bulk pretraining at
$s_{\text{orig}} = 4{,}096$ for over $30\text{T}$ tokens, and stretch the
context window to $s_{\text{ext}} = 32{,}768$ or $131{,}072$ only during the
short mid-training stage.

Why does a model trained at $s_{\text{orig}} = 4{,}096$ fail if you feed it
token $m = 10{,}000$ at test time? Recall from Lecture 01 that Rotary Positional
Embeddings, or RoPE, group each head's $d_{\text{head}}$ channels into 2D pairs
$j \in \{0, \dots, d_{\text{head}}/2 - 1\}$ and rotate pair $j$ at position $m$
by angle $m \theta_j$.

$$\theta_j = \text{base}^{-2j / d_{\text{head}}}, \qquad \lambda_j = \frac{2\pi}{\theta_j} = 2\pi \cdot \text{base}^{2j / d_{\text{head}}}$$

When $m$ goes past $s_{\text{orig}} = 4{,}096$, the slow, long-wavelength pairs
near $j \to d_{\text{head}}/2$ hit rotation angles $m \theta_j$ they never saw
during pretraining. That scrambles the dot products $q_m^\top k_n$.

Stretching every frequency equally by $\theta_j' = \theta_j / s$, where
$s = s_{\text{ext}} / s_{\text{orig}}$, keeps angles in range. Yet it also
squashes adjacent tokens $s\times$ closer together. That hurts the fast,
short-wavelength pairs with $\lambda_j \ll s_{\text{orig}}$, which already
completed many full $2\pi$ turns during pretraining and only track local word
order.

Mid-training therefore combines two targeted adjustments. First, **Adjusted Base
Frequency** raises the RoPE base from $10{,}000$ to $1{,}000{,}000$, as in
Llama 3 and Qwen3, or scales it by the NTK-aware factor
$\text{base}' = \text{base} \cdot s^{d_{\text{head}} / (d_{\text{head}} - 2)}$.
Second, **YaRN** from Peng et al. (2023) splits the frequency pairs by
wavelength. It leaves high-frequency local pairs untouched with $\gamma_j = 1$,
linearly interpolates slow global pairs with $\gamma_j = 0$, and blends the
middle pairs with a smooth ramp $\gamma_j \in [0, 1]$.

$$\theta_j^{\text{YaRN}} = (1 - \gamma_j) \frac{\theta_j}{s} + \gamma_j \theta_j$$

Fixing the angles leaves one more problem. When a query attends over $s\times$
more keys, the softmax denominator $\sum_{k=1}^m \exp(q_m^\top k_k / \sqrt{d_{\text{head}}})$
sums $s\times$ as many positive terms, which flattens the attention weights into
a blurry uniform distribution. YaRN sharpens the distribution back up by
multiplying the pre-softmax logits by an **attention temperature factor**.

$$\sqrt{\frac{1}{t}} = 0.1 \ln(s) + 1$$

For `Qwen3-0.6B` extending $8\times$ from $4{,}096$ to $32{,}768$, the logit
multiplier is $\sqrt{1/t} = 0.1 \ln(8) + 1 = 1.2079$, which corresponds to
temperature $t = 1 / 1.2079^2 = 0.6853$.

---

## What you should see

Running `book/code/scaling_laws.py` shows why Chinchilla's $D \approx 20N$ rule
only applies when inference is free. Once you add lifetime serving FLOPs
$2 N D_{\text{infer}}$, shrinking $N$ and over-training on $300\times$ to
$60{,}000\times$ more tokens cuts total compute in half while making every
decode step faster. During that long training run, $B_{\text{crit}}$ sets the
ceiling on useful data parallelism, WSD saves the final $10\%$ to $20\%$ of
tokens for high-quality annealing, and YaRN stretches the context window without
paying quadratic attention FLOPs from step 1.

---

## Go deeper

- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022), the Chinchilla $L(N, D)$ fit and $20\text{ tokens/param}$ derivation.
- [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) (McCandlish et al., 2018), the gradient noise scale $B_{\text{crit}} = \text{tr}(\Sigma)/\|G\|^2$.
- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) (Yang et al., 2022), the $\mu\text{P}$ width-transfer rules.
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) (Peng et al., 2023), wavelength-band interpolation and attention temperature scaling.

---

## Check yourself

1. Starting from $L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$ and $6ND = C$,
   show why $\alpha = \beta$ implies $N^* \propto C^{0.5}$ and $D^* \propto C^{0.5}$.
2. Why does the critical batch size $B_{\text{crit}} = \text{tr}(\Sigma)/\|G\|^2$
   increase as loss falls, and how do training pipelines exploit that shift?
3. What goes wrong if you stop a Cosine Decay run halfway through, and how does
   a WSD schedule avoid that problem?
4. Why does YaRN leave high-frequency RoPE dimensions unscaled ($\gamma_j = 1$)
   while scaling low-frequency dimensions by $1/s$?

---

## Next

**[31. Activation memory and recomputation](31-activation-memory.md)**:
Derive every byte saved per transformer layer for backward, why eager attention
explodes as $O(s^2)$, why the final vocabulary logits can be the largest tensor
in VRAM, and how selective checkpointing trades $33\%$ extra compute for an
$18\times$ activation memory cut.

```bash
uv run python book/code/activation_budget.py
```

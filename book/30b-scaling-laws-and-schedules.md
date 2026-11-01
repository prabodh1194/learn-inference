# Lecture 30b: Scaling Laws, Learning Rate Schedules & Mid-Training

[Lecture 30](30-training-arithmetic.md) established that one training token through a dense model of $N$ parameters costs $6N$ FLOPs, whereas one inference token costs $2N$ FLOPs. That $3:1$ ratio per token hides a much larger asymmetry over a model's life: a model is trained once on $D_{\text{train}}$ tokens, then served to millions of users across $D_{\text{infer}}$ tokens.

This lecture derives the mathematics of how frontier labs allocate parameters $N$, training tokens $D_{\text{train}}$, batch size $B$, and learning rate schedules $\eta(t)$ across pretraining and **mid-training** (the annealing and context-extension stage between raw web pretraining and post-training).

---

### 1. Chinchilla Compute-Optimal Scaling

Hoffmann et al. (2022) modeled the pretraining cross-entropy loss $L(N, D)$ after training a model of $N$ parameters on $D$ tokens as the sum of an irreducible entropy $E$ of natural text and two power-law approximation errors:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

Empirical fits on the Chinchilla sweep yielded $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, and $\beta = 0.28$.

Suppose you have a fixed pretraining compute budget $C = 6ND$ FLOPs and zero concern for inference cost. What choice of $(N^*, D^*)$ minimizes $L(N, D)$ subject to $6ND = C$?

#### Full Derivation via Substitution

Solve the constraint $6ND = C$ for $D$:

$$D = \frac{C}{6N}$$

Substitute $D = \frac{C}{6N}$ into $L(N, D)$:

$$L(N) = E + A N^{-\alpha} + B \left(\frac{C}{6N}\right)^{-\beta} = E + A N^{-\alpha} + B \left(\frac{C}{6}\right)^{-\beta} N^{\beta}$$

Differentiate $L(N)$ with respect to $N$ and set $\frac{dL}{dN} = 0$:

$$\frac{dL}{dN} = -\alpha A N^{-\alpha - 1} + \beta B \left(\frac{C}{6}\right)^{-\beta} N^{\beta - 1} = 0$$

Multiply both sides by $N^{\alpha + 1}$:

$$\beta B \left(\frac{C}{6}\right)^{-\beta} N^{\alpha + \beta} = \alpha A$$

Solve for $N^{\alpha + \beta}$:

$$N^{\alpha + \beta} = \frac{\alpha A}{\beta B} \left(\frac{C}{6}\right)^{\beta}$$

Raise both sides to the power $\frac{1}{\alpha + \beta}$:

$$N^*(C) = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha + \beta}} \left(\frac{C}{6}\right)^{\frac{\beta}{\alpha + \beta}}$$

By symmetry, substituting $N^*(C)$ back into $D^*(C) = \frac{C}{6 N^*(C)}$ gives:

$$D^*(C) = \left(\frac{\beta B}{\alpha A}\right)^{\frac{1}{\alpha + \beta}} \left(\frac{C}{6}\right)^{\frac{\alpha}{\alpha + \beta}}$$

Plugging in the empirical exponents $\alpha = 0.34$ and $\beta = 0.28$ ($\alpha + \beta = 0.62$):

$$\frac{\beta}{\alpha + \beta} = \frac{0.28}{0.62} \approx 0.452, \qquad \frac{\alpha}{\alpha + \beta} = \frac{0.34}{0.62} \approx 0.548$$

When balanced ($\alpha \approx \beta$), both $N^*$ and $D^*$ scale as $C^{0.5}$, yielding the famous **Chinchilla rule of thumb**:

$$\frac{D^*}{N^*} \approx 20 \text{ tokens per parameter}$$

Under Chinchilla scaling, our reference model `Qwen3-0.6B` ($N = 6 \times 10^8$) should be trained on only:

$$D^* = 20 \times (6 \times 10^8) = 1.2 \times 10^{10} \text{ tokens } (12\text{B tokens})$$

Yet the Qwen3 technical report states that `Qwen3-0.6B` was trained on **$36\text{ trillion}$ tokens** ($3.6 \times 10^{13}$) — exactly **$3{,}000\times$ the Chinchilla ratio** ($60{,}000$ tokens per parameter). Why?

---

### 2. Inference-Aware Scaling Laws: Why Modern Models Over-Train

Chinchilla optimizes $L(N, D_{\text{train}})$ under a fixed *training* budget $C_{\text{train}} = 6 N D_{\text{train}}$. In production, you pay for both training and lifetime serving across $D_{\text{infer}}$ generated tokens:

$$C_{\text{total}}(N, D_{\text{train}}) = \underbrace{6 N D_{\text{train}}}_{\text{paid once}} + \underbrace{2 N D_{\text{infer}}}_{\text{paid across all users}}$$

Suppose you want a model that achieves a target validation loss $L_{\text{target}}$ while minimizing **lifetime compute** $C_{\text{total}}$. Let us compare three ways to reach low loss when serving $D_{\text{infer}} = 100\text{T}$ ($10^{14}$) lifetime tokens:

1. **Chinchilla $2.8\text{B}$ ($20\text{ tok/param}$):** $N = 2.8 \times 10^9$, $D_{\text{train}} = 5.6 \times 10^{10}$ ($56\text{B}$ tokens).
   - Loss: $L = 1.69 + 406.4 (2.8 \times 10^9)^{-0.34} + 410.7 (5.6 \times 10^{10})^{-0.28} = 1.69 + 0.250 + 0.401 = 2.341$.
   - Training FLOPs: $6 (2.8 \times 10^9)(5.6 \times 10^{10}) = 9.41 \times 10^{20} \approx 0.9\text{ ZFLOP}$.
   - Inference FLOPs: $2 (2.8 \times 10^9)(10^{14}) = 5.60 \times 10^{23} = 560.0\text{ ZFLOP}$.
   - Total lifetime compute: **$560.9\text{ ZFLOP}$**.
2. **Over-trained $1.4\text{B}$ ($300\text{ tok/param}$):** $N = 1.4 \times 10^9$, $D_{\text{train}} = 4.2 \times 10^{11}$ ($420\text{B}$ tokens).
   - Loss: $L = 1.69 + 406.4 (1.4 \times 10^9)^{-0.34} + 410.7 (4.2 \times 10^{11})^{-0.28} = 1.69 + 0.316 + 0.228 = 2.234$ (better than Chinchilla $2.8\text{B}$!).
   - Training FLOPs: $6 (1.4 \times 10^9)(4.2 \times 10^{11}) = 3.53 \times 10^{21} \approx 3.5\text{ ZFLOP}$.
   - Inference FLOPs: $2 (1.4 \times 10^9)(10^{14}) = 280.0\text{ ZFLOP}$.
   - Total lifetime compute: **$283.5\text{ ZFLOP}$** ($1.98\times$ cheaper overall, plus half the HBM bandwidth per decode step).
3. **`Qwen3-0.6B` scale ($60{,}000\text{ tok/param}$):** $N = 0.6 \times 10^9$, $D_{\text{train}} = 3.6 \times 10^{13}$ ($36\text{T}$ tokens).
   - Loss: $L = 1.69 + 406.4 (0.6 \times 10^9)^{-0.34} + 410.7 (3.6 \times 10^{13})^{-0.28} = 1.69 + 0.421 + 0.066 = 2.177$.
   - Training FLOPs: $6 (0.6 \times 10^9)(3.6 \times 10^{13}) = 129.6\text{ ZFLOP}$.
   - Inference FLOPs: $2 (0.6 \times 10^9)(10^{14}) = 120.0\text{ ZFLOP}$.
   - Total lifetime compute: **$249.6\text{ ZFLOP}$**, and the model fits in $1.2\text{ GB}$ of BF16 weights on an edge device.

??? question "Why did Chinchilla say 20 tokens per parameter while Llama 3 and Qwen3 train on 1,000 to 60,000 tokens per parameter?"
    Chinchilla answered a narrow question: *given a fixed pretraining budget $C_{\text{train}}$, what $(N, D)$ minimizes pretraining loss?* It assigned zero cost to inference. In practice, inference dominates lifetime fleet cost ($C_{\text{infer}} = 2 N D_{\text{infer}}$) and decode latency is bound by reading $2N$ bytes of weights from HBM at every token ([Lecture 02](02-arithmetic-intensity.md)). Spending $4\times$–$100\times$ more FLOPs once during pretraining to shrink $N$ by $2\times$–$4\times$ pays back permanently on every generated token. See [Q&A: Why did Chinchilla say 20 tokens per parameter while Llama 3 and Qwen3 train on 1,000+ tokens per parameter?](qa.md#why-did-chinchilla-say-20-tokens-per-parameter-while-llama-3-and-qwen3-train-on-1000-tokens-per-parameter).

---

### 3. Critical Batch Size and Batch-Size Ramp

Why can we not simply parallelize those $36\text{T}$ training tokens across a million GPUs by making the global batch size $B$ arbitrarily large?

McCandlish et al. (2018) showed that stochastic gradients $g_i = \nabla_\theta \ell(x_i)$ have true mean $G = \mathbb{E}[g_i]$ and covariance $\Sigma = \text{Cov}(g_i)$. Averaging over a batch of $B$ independent samples produces a batch gradient $\bar{g}_B$ with expectation $G$ and covariance $\frac{1}{B}\Sigma$. To second order, the expected improvement in loss from a step along $\bar{g}_B$ saturates once the noise term $\frac{1}{B}\text{tr}(H\Sigma)$ falls below the signal $\|G\|_H^2$. The transition occurs at the **gradient noise scale** (or **critical batch size**):

$$B_{\text{crit}} \approx B_{\text{noise}} = \frac{\text{tr}(\Sigma)}{\|G\|^2}$$

Calling $S_{\min}$ the minimum number of serial optimizer steps needed at $B \to \infty$ and $E_{\min} = B_{\text{crit}} S_{\min}$ the minimum number of training tokens needed as $B \to 0$, the number of steps $S(B)$ and total tokens $E(B) = B \cdot S(B)$ required to reach a target loss obey the trade-off:

$$\frac{S(B)}{S_{\min}} = 1 + \frac{B_{\text{crit}}}{B}, \qquad \frac{E(B)}{E_{\min}} = 1 + \frac{B}{B_{\text{crit}}}$$

- **Small batch regime ($B \ll B_{\text{crit}}$):**Doubling $B$ halves the required steps $S(B)$ with almost zero extra token waste ($E(B) \approx E_{\min}$). Because gradient variance scales as $1/B$, the optimal learning rate scales as $\eta \propto B$ (or $\sqrt{B}$ for Adam).
- **Large batch regime ($B \gg B_{\text{crit}}$):** The batch gradient is already noiseless ($\bar{g}_B \approx G$). Doubling $B$ yields zero reduction in steps ($S(B) \approx S_{\min}$) and **doubles the wasted tokens** ($E(B) \propto B$).
- **Why $B_{\text{crit}}$ grows during training:** Early in training, the true gradient $\|G\|$ is large, so $B_{\text{crit}} = \text{tr}(\Sigma)/\|G\|^2$ is small. As the model approaches a minimum, $\|G\|$ shrinks while $\text{tr}(\Sigma)$ stays roughly constant, causing $B_{\text{crit}}$ to grow by $10\times$–$20\times$. This is why DeepSeek-V3 and Llama 3 **ramp the global batch size** from $4\text{M}$ tokens early in training to $16\text{M}$–$60\text{M}$ tokens later in training.

---

### 4. Warmup-Stable-Decay (WSD) vs. Cosine Schedules & $\mu\text{P}$

Classic pretraining uses a **Cosine Decay** learning-rate schedule after $T_w$ warmup steps over a predetermined horizon $T_{\max}$:

$$\eta_{\text{cos}}(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\pi \frac{t - T_w}{T_{\max} - T_w}\right)\right)$$

Cosine decay has a severe operational flaw: **you must commit to $T_{\max}$ before step 1**. You cannot stop early at $0.5\,T_{\max}$ (the learning rate is still high, so weights oscillate around the valley floor) and you cannot continue smoothly past $T_{\max}$.

Modern frontier pretraining (MiniCPM, DeepSeek-V3, OLMo 2, Llama 3) replaces Cosine with **Warmup-Stable-Decay (WSD)**, also called ** trapezoidal scheduling**:

$$\eta_{\text{WSD}}(t) = \begin{cases}
\eta_{\max} \frac{t}{T_w} & 0 \le t \le T_w \quad \text{(Warmup, }\sim 2{,}000\text{ steps)} \\[4pt]
\eta_{\max} & T_w < t \le T_s \quad \text{(Stable plateau, }80\%\text{–}90\%\text{ of tokens)} \\[4pt]
\eta_{\max} \left(1 - \frac{t - T_s}{T_{\max} - T_s}\right) & T_s < t \le T_{\max} \quad \text{(Rapid Decay / Annealing, }10\%\text{–}20\%\text{ of tokens)}
\end{cases}$$

WSD separates exploration from convergence:

1. The **Stable phase** holds $\eta = \eta_{\max}$ indefinitely without specifying $T_{\max}$ in advance.
2. At any checkpoint ($1\text{T}$, $5\text{T}$, or $30\text{T}$ tokens), you can branch off a **Decay (Annealing) phase** over the final $10\%$ of tokens while swapping the data mixture from noisy web crawl to high-quality STEM, math, and code (**mid-training**).
3. To set $\eta_{\max}$ without sweeping at $70\text{B}+$ scale, labs use **Maximal Update Parametrization ($\mu\text{P}$, Yang et al. 2022)**: if hidden weights initialize with variance $1/d_{\text{model}}$ and their Adam learning rate scales as $\eta_{\text{hidden}} = \eta_{\text{base}} \cdot (d_{\text{proxy}} / d_{\text{target}})$ while input embeddings keep $\eta_{\text{embed}} = \eta_{\text{base}}$, every layer's feature activation change $\|\Delta h_\ell\|_2 / \sqrt{d_{\text{model}}}$ stays $\Theta(1)$ across widths, allowing optimal hyperparameters tuned on a $40\text{M}$ proxy model to transfer zero-shot to $70\text{B}+$.

---

### 5. Mid-Training Long-Context Extension: RoPE Scaling & YaRN

Pretraining at $s = 32{,}768$ or $131{,}072$ for all $36\text{T}$ tokens would inflate quadratic attention FLOPs by $8\times$–$32\times$. Instead, models like `Qwen3` train Stage 1 and Stage 2 at $s_{\text{orig}} = 4{,}096$ for $30\text{T}+$ tokens, and extend context to $s_{\text{ext}} = 32{,}768$ or $131{,}072$ only during the final **mid-training** stage.

Recall from [Lecture 01](01-the-two-phases.md) that Rotary Positional Embeddings (RoPE) rotate each 2D pair $j \in \{0, \dots, d_{\text{head}}/2 - 1\}$ at position $m$ by angle $m \theta_j$, where:

$$\theta_j = \base^{-2j / d_{\text{head}}}, \qquad \text{with wavelength } \lambda_j = \frac{2\pi}{\theta_j} = 2\pi \cdot \base^{2j / d_{\text{head}}}$$

When $m$ exceeds $s_{\text{orig}} = 4{,}096$, low-frequency dimensions ($j \to d_{\text{head}}/2$) encounter rotation angles $m \theta_j$ never seen during pretraining, destroying attention scores. Two mechanisms fix this during mid-training:

1. **Adjusted Base Frequency (ABF) / NTK-Aware Scaling:** Increasing the RoPE base from $\base = 10{,}000$ to $\base' = 1{,}000{,}000$ (used in Llama 3 and Qwen3) stretches wavelengths $\lambda_j$, or equivalently scales the base by the NTK factor $\base' = \base \cdot s^{\frac{d_{\text{head}}}{d_{\text{head}}-2}}$ for extension ratio $s = s_{\text{ext}} / s_{\text{orig}}$.
2. **YaRN (Yet another RoPE extensioN, Peng et al. 2023):** High-frequency dimensions ($\lambda_j \ll s_{\text{orig}}$) have already completed many full $2\pi$ periods during pretraining and only encode *relative* local distance; interpolating them blurs local token order. YaRN leaves high frequencies untouched ($\gamma_j = 1$), linearly interpolates low frequencies ($\gamma_j = 0 \implies \theta_j' = \theta_j / s$), and blends intermediate frequencies via a ramp $\gamma_j \in [0, 1]$:

$$\theta_j^{\text{YaRN}} = (1 - \gamma_j) \frac{\theta_j}{s} + \gamma_j \theta_j$$

Furthermore, as sequence length grows by factor $s$, the softmax partition function $\sum_{k=1}^m \exp(q_m^\top k_k / \sqrt{d_{\text{head}}})$ sums over $s\times$ more keys, flattening the attention distribution's entropy. YaRN counteracts this entropy growth by multiplying the pre-softmax logits (or scaling $Q$ and $K$) by an **attention temperature scale**:

$$\sqrt{\frac{1}{t}} = 0.1 \ln(s) + 1$$

For `Qwen3-0.6B` extending from $4{,}096 \to 32{,}768$ ($s = 8$):

$$\sqrt{\frac{1}{t}} = 0.1 \ln(8) + 1 = 0.1(2.0794) + 1 = 1.2079 \implies t = \frac{1}{1.2079^2} = 0.6853$$

---

### Running the Scaling & YaRN Calculator

[`book/code/scaling_laws.py`](code/scaling_laws.py) computes the Chinchilla vs. over-trained lifetime FLOP ledger and YaRN temperature scaling factors:

```bash
uv run python book/code/scaling_laws.py
```

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

---

### Key Takeaways

- **Chinchilla ($D \approx 20N$)** minimizes pretraining FLOPs alone; **inference-aware scaling** minimizes $6ND_{\text{train}} + 2ND_{\text{infer}}$, justifying $1{,}000\times$–$3{,}000\times$ over-training (`Qwen3-0.6B` on $36\text{T}$ tokens).
- **Critical batch size $B_{\text{crit}} = \text{tr}(\Sigma)/\|G\|^2$** bounds data-parallel scaling: increasing $B$ beyond $B_{\text{crit}}$ wastes tokens linearly without reducing serial steps.
- **Warmup-Stable-Decay (WSD)** decouples training horizon selection from step 1 and reserves the final $10\%$–$20\%$ decay window for mid-training data annealing and YaRN long-context extension.

---

*Next: [Lecture 31: Activation Memory & Selective Checkpointing](31-activation-memory.md)*

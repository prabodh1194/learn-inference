# 36. Supervised fine-tuning (SFT), LoRA, and QLoRA

**Demo:** `book/code/lora_memory.py` · **Build:** `train/sft_lora.py` · **Test:** `tests/test_36_sft_lora.py`
**Moves:** trainable parameters down $59\times$ ($596\text{M} \to 10.1\text{M}$); optimizer + gradient memory down $98.3\%$
**Prereq:** [12b. Structured output and adapters](12b-structured-output.md), [30. The training ledger](30-training-arithmetic.md), [31. Activations and checkpointing](31-activation-memory.md)

---

## The problem

A pre-trained base model (`Qwen3-0.6B-Base`) is a document completer: ask it
`"What is the capital of France?"` and it is just as likely to continue with
`"What is the capital of Germany? What is the capital of Spain?"` (because exam
worksheets in Common Crawl look like lists of questions).

**Supervised Fine-Tuning (SFT)** teaches the model to act as an assistant by
training on `(prompt, response)` pairs formatted with chat templates
(`<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n...\n<|im_end|>`).

Two questions immediately arise:
1. **What changes in the loss function between pre-training and SFT?** Only one
   thing — and getting it wrong wastes most of your gradient budget memorizing
   user prompts.
2. **Why pay Lecture 30's $16\Psi$ memory bill (8.88 GiB on 0.6B, 104 GiB on 7B)
   to adjust a model that already knows English, code, and math?** **LoRA** and
   **QLoRA** freeze the base weights and train a low-rank update — but with a
   crucial activation-memory catch that catches almost everyone off guard.

---

## See it

```bash
uv run python book/code/lora_memory.py
```

```
Qwen3-0.6B (596.0M total, LoRA r=16 on all 7 projections = 10.09M trainable, 1.69%)
=========================================================================
  mode         base weights   grads+AdamW   activations(B=1,S=2048)    total
  Full FT        1136.9 MiB    7958.1 MiB               1904.0 MiB   10.74 GiB
  LoRA (bf16)    1136.9 MiB     134.8 MiB               1904.0 MiB    3.10 GiB
  QLoRA (nf4)     312.6 MiB     134.8 MiB               1904.0 MiB    2.30 GiB

  Notice: LoRA crushes grads+AdamW by 98%, so ACTIVATIONS become the dominant term!
```

Read the three columns across the rows:
- Full fine-tuning spends **7,958.1 MiB** ($14\Psi$: $2\Psi$ BF16 gradients +
  $12\Psi$ FP32 AdamW state) on trainable state.
- **LoRA ($r = 16$)** cuts that $7{,}958.1\text{ MiB}$ to **134.8 MiB** ($59\times$
  smaller!), and **QLoRA** shrinks the frozen base weights from **1,136.9 MiB to
  312.6 MiB**.
- Yet the third column — **1,904.0 MiB of activations** — does **not move at all**!

Let's derive every number in that table, and why the third column refuses to budge.

---

## The idea

### 1. SFT loss masking: why prompt tokens get `labels = -100`

In pre-training, every token in the sequence is both an input context for later
tokens and a training target. In SFT, a training example has two parts:
- Prompt tokens $x = (x_1, \dots, x_P)$ (system prompt, user question, tool output)
- Assistant response tokens $y = (y_1, \dots, y_R)$ (ending with `<|im_end|>`)

We want to maximize the conditional log-likelihood of the **response given the
prompt**:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{R} \sum_{t=1}^{R} \log \pi_\theta(y_t \mid x_1, \dots, x_P, y_1, \dots, y_{t-1})$$

The forward pass still runs over the concatenated sequence $[x_1, \dots, x_P, y_1, \dots, y_R]$
of length $P + R$ (so the response tokens can attend to the prompt's K and V!),
but the label vector sets every prompt position to PyTorch's `ignore_index = -100`:

```
input_ids : [ <user>,  "2",  "+",  "2", "?", <asst>,  "4", <eos> ]
labels    : [   -100, -100, -100, -100, -100,  -100,  "4", <eos> ]
              └──────────── prompt (masked) ──────────┘ └─ scored ─┘
```

What happens if you forget to mask the prompt (`labels = input_ids.clone()`)?
Suppose a rag/coding prompt has $P = 2{,}000$ tokens of context and a $R = 20$
token answer. Without prompt masking, $\frac{2000}{2020} = 99\%$ of the gradient
updates the model to predict the user's messy prompt rather than the assistant's
answer!

**Multi-turn masking & SFT hyperparameter regimes.** In a multi-turn conversation
$(u_1, a_1, u_2, a_2, \dots, u_K, a_K)$, two conventions exist:
- **Mask user/tool turns only:** Every assistant reply $a_1, \dots, a_K$ receives
  supervision (`labels != -100`) in a single forward pass, while all `system`,
  `user`, and `tool` observation tokens are set to `-100`.
- **Final-turn only (unrolled):** Unroll a $K$-turn dialogue into $K$ examples
  where example $k$ supervises only $a_k$ and masks $(u_1, a_1, \dots, u_k)$.
- **Why SFT uses $10\times$–$30\times$ smaller learning rates and batches:** To
  avoid overwriting pre-trained representations on a $1\text{M}$-prompt SFT mix,
  models like `OLMo 2` drop the peak learning rate from $3 \times 10^{-4}$
  (pre-training) to $1 \times 10^{-5}$–$5 \times 10^{-5}$ (SFT) and reduce global
  batch size from $4\text{M}$ tokens to $256$ prompts ($1$–$2$ epochs total).

---

### 2. LoRA math: parameter count, initialization, and $\alpha / r$ scaling

In Lecture 12b, we met LoRA from the serving side. Now derive it from the
training side (Hu et al., 2021).

For a frozen pre-trained weight matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$,
LoRA parameterizes the adapted weight matrix as:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$

where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$
are trainable low-rank matrices with rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$,
and $\alpha$ is a constant hyperparameter (typically $\alpha = r$ or $\alpha = 2r$).

#### Exact parameter count on Qwen3-0.6B ($r = 16$)

Each of the $L = 28$ layers in Qwen3-0.6B has 7 linear projections. Count the
trainable parameters $(d_{\text{in}} + d_{\text{out}}) \times r$ for each:

```
Q_proj    (1024 -> 2048):  16 × (1024 + 2048)  =   49,152 params
K_proj    (1024 -> 1024):  16 × (1024 + 1024)  =   32,768 params
V_proj    (1024 -> 1024):  16 × (1024 + 1024)  =   32,768 params
O_proj    (2048 -> 1024):  16 × (2048 + 1024)  =   49,152 params
gate_proj (1024 -> 3072):  16 × (1024 + 3072)  =   65,536 params
up_proj   (1024 -> 3072):  16 × (1024 + 3072)  =   65,536 params
down_proj (3072 -> 1024):  16 × (3072 + 1024)  =   65,536 params
----------------------------------------------------------------
per layer total                                =  360,448 params
all 28 layers:  28 × 360,448                   = 10,092,544 params (10.09M = 1.69% of 596.0M)
```

Now plug $\Psi_{\text{LoRA}} = 10{,}092{,}544$ into Lecture 30's $14\Psi$
gradient + AdamW state formula ($2\text{ B}$ BF16 grads + $12\text{ B}$ FP32
master weights and moments):

$$\text{Bytes}_{\text{LoRA grads+opt}} = 14 \times 10{,}092{,}544 = 141{,}295{,}616\text{ B} = 134.75\text{ MiB}$$

against $14 \times 596{,}049{,}920 = 7{,}958.1\text{ MiB}$ for full fine-tuning —
the exact **134.8 MiB vs 7,958.1 MiB** in the demo!

#### Why $B = 0$ and $A \sim \mathcal{N}(0, \sigma^2)$ at initialization

Look at the forward and backward equations of a LoRA layer
$Y = X W_0^\top + \frac{\alpha}{r} (X A^\top) B^\top$:

$$\nabla_B = \frac{\alpha}{r} \nabla_Y^\top (X A^\top), \qquad \nabla_A = \frac{\alpha}{r} (B^\top \nabla_Y^\top)^\top X$$

Check the three possible initialization choices at step $0$:
1. **Both $A \neq 0$ and $B \neq 0$:** then $\Delta W = \frac{\alpha}{r} B A \neq 0$
   at step 0, injecting random noise into every layer of your pre-trained model
   before a single gradient step has been taken!
2. **Both $A = 0$ and $B = 0$:** then $\Delta W = 0$ at step 0, **but look at
   the gradients:**
   $$\nabla_B = \frac{\alpha}{r} \nabla_Y^\top (X \cdot 0) = 0, \qquad \nabla_A = \frac{\alpha}{r} (0 \cdot \nabla_Y^\top)^\top X = 0$$
   Both gradients are identically zero! It is a saddle point; neither $A$ nor $B$
   can ever move!
3. **$B = 0$ and $A \sim \mathcal{N}(0, \sigma^2)$ (the LoRA rule):**
   - At step 0, $\Delta W = \frac{\alpha}{r} (0 \cdot A) = 0$, so the model starts
     **bit-for-bit identical** to the pre-trained base model.
   - At step 0, $\nabla_A = 0$, **but $\nabla_B = \frac{\alpha}{r} \nabla_Y^\top (X A^\top) \neq 0$**
     because $A \neq 0$! So $B$ becomes non-zero on step 1, and from step 2
     onward both $\nabla_A$ and $\nabla_B$ are non-zero!

---

### 3. Why LoRA does NOT shrink activation memory (and how QLoRA works)

Look again at the backward equation for $\nabla_B$ and $\nabla_A$ in layer $\ell$:

$$\nabla_{B_\ell} = \frac{\alpha}{r} \nabla_{Y_\ell}^\top (X_\ell A_\ell^\top)$$

To compute $\nabla_{B_1}$ in **Layer 1**, you need two things:
1. The forward input $X_1$ to Layer 1 (and $X_\ell$ at every adapted layer
   $\ell = 1, \dots, 28$).
2. The upstream activation gradient $\nabla_{Y_1}$, which only reaches Layer 1
   by backpropagating $\nabla_X = \nabla_Y W_0^\top$ **all the way through the
   frozen base weights $W_0$ of Layers 28 down to 2**!

??? question "Wait — if the base weights $W_0$ are frozen (`requires_grad=False`), why does backpropagation still run through every layer and save activations?"
    Freezing $W_0$ skips only the **weight-gradient** matmul $\nabla_{W_0} = X^\top \nabla_Y$
    for $W_0$ itself. Because LoRA adapters are attached to Layer 1 through Layer
    28, the **input-gradient** $\nabla_X = \nabla_Y W_0^\top$ must still propagate
    from the loss all the way back to Layer 1, and every adapted layer (plus
    every non-linearity like SwiGLU, RMSNorm, and attention along that path)
    must still save its forward activations! That is why LoRA + gradient
    checkpointing is mandatory on long sequences.
    [Full answer](qa.md#if-base-weights-are-frozen-in-lora-why-doesnt-activation-memory-drop)

**QLoRA (Dettmers et al., 2023)** attacks the remaining **1,136.9 MiB** of
frozen base weights $W_0$:
- Store $W_0$ in **4-bit NormalFloat (NF4)** ($0.5\text{ bytes/param}$), a
  quantile-spaced 16-bin lookup table optimized for normally distributed weights.
- **Double quantization:** quantize the FP32 block-scale factors (one per 64
  weights) down to FP8 in groups of 256, cutting scale overhead from
  $32 / 64 = 0.5\text{ bits/param}$ to $8 / 64 + 32 / (64 \times 256) \approx 0.127\text{ bits/param}$
  (total $\approx 0.55\text{ bytes/param} \implies 312.6\text{ MiB}$ on Qwen3-0.6B).
- During forward and backward ($\nabla_X = \nabla_Y W_0^\top$), dequantize $W_0$
  on the fly to BF16 tile-by-tile, while keeping the small LoRA adapters $A, B$
  in BF16/FP32.

---

## The code

```python
def build_sft_labels(
    prompt_ids: list[int], response_ids: list[int], ignore_index: int = -100
) -> list[int]:
    return [ignore_index] * len(prompt_ids) + list(response_ids)


def lora_param_count(d_in: int, d_out: int, rank: int) -> int:
    return rank * (d_in + d_out)


def lora_forward(x, w0, a, b, alpha: float):
    rank = a.shape[0]
    base = x @ w0.T
    adapter = (x @ a.T) @ b.T
    return base + (alpha / rank) * adapter
```

Look at the parentheses in `(x @ a.T) @ b.T`:
- `(x @ a.T)` multiplies $(D, d_{\text{in}}) \times (d_{\text{in}}, r) \to (D, r)$,
  costing $2 D d_{\text{in}} r$ FLOPs.
- Then `@ b.T` multiplies $(D, r) \times (r, d_{\text{out}}) \to (D, d_{\text{out}})$,
  costing $2 D r d_{\text{out}}$ FLOPs.
- If you instead wrote `x @ (a.T @ b.T)`, you would materialize the full
  $(d_{\text{in}}, d_{\text{out}})$ matrix and pay $2 D d_{\text{in}} d_{\text{out}}$
  FLOPs! Associativity of matrix multiplication is a $30\times$ FLOP difference.

---

## Build it

1. Implement `build_sft_labels`, `lora_param_count`, and `lora_forward` in
   `train/sft_lora.py`.
2. Run the tests:

```bash
uv run pytest tests/test_36_sft_lora.py -v
```

3. Verify that when `b = torch.zeros(d_out, r)` and `a = torch.randn(r, d_in)`,
   `lora_forward(x, w0, a, b, alpha=16)` equals `x @ w0.T` to the last bit.

---

## What you should see

- **LoRA reduces gradient + optimizer state by $98.3\%$** on Qwen3-0.6B at $r=16$.
- **Zero-initializing $B$ preserves exact base-model outputs at step 0** while
  giving $B$ a non-zero gradient on step 1.
- **Merging adapters at inference time** ($W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$)
  gives **zero serving latency overhead** for a single adapter, or multi-LoRA
  batching (Lecture 12b) when serving many adapters simultaneously.

---

## Go deeper

- **[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)** (Hu et al., 2021).
- **[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)** (Dettmers et al., 2023) — NF4, double quantization, and paged optimizers.
- **[LoRA Without Regrets](https://thinkingmachines.ai/blog/lora/)** (Schulman et al., Thinking Machines Lab, 2025) — why applying LoRA to **all 7 linear layers** (especially MLP gate/up/down, not just Q and V) matches full fine-tuning in post-training.

---

## Check yourself

1. Why do we set `labels = -100` on prompt tokens during SFT, while still keeping
   those prompt tokens in `input_ids`?
2. What happens to $\nabla_A$ and $\nabla_B$ at step 0 if you initialize **both**
   $A = 0$ and $B = 0$?
3. Why does LoRA save almost all optimizer memory while saving almost **no**
   activation memory?
4. Why is `(x @ a.T) @ b.T` much faster than `x @ (a.T @ b.T)`?

---

## Next

**[36b. Rejection sampling, Best-of-N, and synthetic data engines](36b-rejection-sampling-and-synthetic-data.md)**:
derive the exact $\ln N - \frac{N-1}{N}$ KL cost of Best-of-$N$, compare per-prompt vs. global rejection sampling, and build synthetic data pipelines.

```bash
uv run python book/code/rejection_sampling.py
```

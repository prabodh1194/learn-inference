# Transcript audit 2 — CMU inference course, DeepSeek, vLLM, Reiner Pope

Audited 2026-08-15. Transcripts in `book/assets/transcripts/`; lectures cited as
`B##`; qa.md as `qa:LINE`. This is the second audit pass: audit 1 covered the
attention cluster; audit 2 covers the CMU "Inference Algorithms for Language
Modeling" course (12 lectures), the DeepSeek/V4/vLLM family, the Claude Code
pair, and the two Reiner Pope interviews. All findings were verified against the
book text and, where they landed, merged into the lectures (commits
`9201132`, `32a240d`, `d968704`).

Severity: HIGH = change the book; MED = worth adding (verified, not yet
present); LOW = record only.

---

## 1. Reiner Pope × Dwarkesh — training & serving (`xmkSf5IS-zw`)

**1.1 HIGH — batch-balance rule `B ≈ (F/BW) × (N_total/N_active)`.** Per-token
decode cost is the weight re-read divided by the batch; the compute floor is
`1/(F/BW)`. Balancing gives the batch size where memory stops binding: ~300×
sparsity on datacenter FP4 (B ≈ 2,400 for a 32/256-expert MoE as Pope stated;
the real V3 is 8/256 → ~18× param ratio → B ≈ 5,400). The book's cost model
(L28) had no batch dimension. **Landed** in L28 "The batch floor" + qa entry.

**1.2 HIGH — ~20 ms per-step cadence.** A decode step's floor is
`capacity/bandwidth` (3090: 24 GB/936 GB/s ≈ 26 ms; stays ~20 ms across HBM
generations). A request waits ≤1 cadence to board + 1 to finish ≈ 50 ms floor
under any queueing. **Landed** in L25 collapsible + qa entry.

**1.3 MED — hosted pricing decodes as memory-bound decode.** 3–5× output/input
price ratio is external evidence of memory-bound decode at low batch. **Landed**
in L28 "Predict first".

**1.4 MED — 200k-context crossover reconciles with the book's 8k.** Same balance
point, `bytes/token` and active params differ ~100–3000×. The book's earlier
"~200k" derivation was garbled — see §11 (fixed).

**1.5 MED — pipelining is latency-neutral in inference.** Training bubbles vs
inference KV-invariance (in-flight sequences rise with stages, cancelling the
memory win). **Landed** in L22 PP row + prose.

**1.6 LOW — "~300" is FP4-datacenter-specific.** A100 FP16 gives ~312, H100
FP16 ~590; compute F/BW at your own dtype. **Landed** in L02 (FP4 ≠ half of FP8).

**1.7 LOW — fleet sizing.** One saturated rack ≈ 128k tok/s ≈ 1/1000th of
Gemini-scale traffic.

**1.8 LOW — memory-tier "drain time"** (capacity/bandwidth) explains 5-min vs
1-hour cache-write retention pricing.

## 2. Reiner Pope × Dwarkesh — chip design (`oIk3R-sMX5o`)

**2.1 MED — the "~100×" SRAM figure is CPU cache-vs-DDR *latency*, not GPU
bandwidth.** 3090 aggregate shared-memory bandwidth ≈ 19 TB/s vs 936 GB/s HBM
≈ 20×. Resolves the book's hedge. **Landed** in qa.md SRAM note.

**2.2 MED — inside the chip, feeding arithmetic costs ~6× the arithmetic** (3
register-file muxes ≈ 24p ANDs vs a 4-bit MAC ≈ 4p). The L02 theme one level
down.

**2.3 MED — FP4 ≠ half of FP8: quadratic bit-width scaling** (Dadda multiplier
area is p×q partial products, so 8→4 bits is 4× area, shipped as 3× on B300).
**Landed** in L02 FLOPS box.

**2.4 MED — "a GPU is a lot of tiny TPUs"; cache vs scratchpad is a determinism
choice, not a speed difference.** Shared memory needs explicit instructions
because it's software-controlled.

**2.5 LOW — batch-1-vs-batch-1000 framing** confirms the book's latency-vs-
throughput claim.

**2.6 LOW — FPGA-vs-ASIC economics** (~10×, $10k vs $30M tape-out), hardware
background for L22/L28.

## 3. CMU 1+2 — intro & probability review (`F-mduXzNcRQ`, `UKuPCxozypU`)

**3.1 HIGH — "greedy is deterministic" stated too absolutely.** fp reduction
order is nondeterministic across devices/kernels; a tie-adjacent argmax can
flip. The book's oracle claim is right *for a fixed device*. **Landed** in L03
caveat + qa entry.

**3.2 MED — search error vs model error** taxonomy absent from the book.
**Landed** in glossary.

**3.3 MED — T=1 is the only unbiased temperature** (tempered distribution
`q ∝ p^(1/T)` otherwise). **Landed** in L06 + qa entry.

**3.4 MED — chain rule is *why* causal masking holds; shorter sequences are
higher probability; length-normalized log-prob for comparing generations.**
Glossary/qa material.

**3.5 LOW —** Llama 3.1 table verified (8B/70B/405B; MLP ratio ~3.5×), GQA also
cuts K/V *projection* compute, stop conditions are a design choice. GPT-2
"vocab of 5,000" is a garble of 50,257.

## 4. CMU 3 — common sampling methods (`fvbR-9OXUvo`)

**4.1 MED — the greedy-degeneration mechanism is absent.** P(100 heads) =
0.6¹⁰⁰ ≈ 6.5e-23 but a typical 60/40 string ~5e-30 × C(100,60) ≈ 1.4e28:
argmax chases a degenerate spike. **Landed** in L06 collapsible + qa entry.

**4.2 MED — epsilon / η / min-p / locally-typical sampler family absent.**
min-p is *not* in L06 (premise correction). **Landed** in L06 Go-deeper + qa.

**4.3 MED — top-k coverage numbers (68% vs 99% at the same k)** quantify the
top-p motivation.

**4.4 MED — locally typical sampling** (sort by |log p − H|, top-p the result)
— the book omits entropy-based truncation entirely.

**4.5 MED — sampling defaults are per-model `generation_config.json`, not HF's
library defaults.** HF ships do_sample=False, T=1, top_p=1, top_k=50; Llama 3.1
ships T=0.6/top_p 0.9. Caption garble "temperature is 6" = 0.6.

**4.6 MED — perplexity = 2^CE never defined in the book.** **Landed** in glossary.

**4.7 MED — calibration and RLHF damaging it**; post-RLHF logits are a ranking,
not a confidence.

**4.8 MED — model-error vs search-error as the sampler-choice rubric.**

**4.9 LOW —** top-p "seed with top token then add" guard; STT garble ("being
erotic" = ergodic, "murat decoding" = Mirostat).

## 5. CMU 4+5 — beam search & A* (`2hhyfPYGCmY`, `Cal4oRoumTw`)

**5.1 HIGH — the book never answers "why greedy/sampling instead of beam
search?"** Frontier serving has dropped it; beam targets the likelihood-trap
mode. **Landed** in L06 collapsible + qa entry.

**5.2 HIGH — beam width K *is* the batch axis.** K beams run as one batch-K
forward pass: weight traffic ~unchanged (memory-bound), real cost is ×K KV
cache (112 KiB × ctx × K). **Landed** in L06 beam collapsible + qa.

**5.3 MED — per-token mode vs sequence mode** (argmax is the per-token mode;
the sequence argmax over V^T is search-hard).

**5.4 MED — Gumbel-Max trick** (verified: P(argmax(z+G₁,z+G₂)=1) = softmax) —
the "sample without softmax" route.

**5.5 MED — length normalization** (raw path score is non-increasing; divide by
|y|^α; HF uses ((5+len)/6)^α). **Landed** in L06b lecture.

**5.6 MED — repetition trap / likelihood trap mechanism** (Holtzman) behind the
book's "greedy repeats".

**5.7 LOW — "curse of beam search"** (Meister et al.): bigger beam, worse
top-1. "depth first search" is a garble of "best-first".

## 6. CMU 6+8 — controlled generation & self-refine (`i4COjX4z1zY`, `uaxf9yssDy4`)

**6.1 HIGH — token healing absent from L12b.** Masking forces legal-but-
low-mass tokenizations ("http"+":"+"//" vs the "://" token); healing rolls back
one token and filters by prefix. **Landed** in L12b.

**6.2 HIGH — FSM vs pushdown expressiveness.** Unbounded nesting needs a stack
(that stack is per-sequence state surviving preemption); "all keys unique" is
unenforceable by masking. **Landed** in L12b + qa entry.

**6.3 HIGH — FUDGE / future discriminators** (log-space factorisation:
`log p ∝ log p(x|prefix) + log p(a|prefix+x)`) — the missing middle between
masking and fine-tuning. Contrastive/adversarial decoding = 2× compute (two
forwards/step), the anti-repetition sibling of L06's penalty.

**6.4 MED — masking preserves relative ranking within the legal set** (proof:
`Q(a)/Q(b) = P(a)/P(b)`; the only distortion is the support). **Landed** in qa
entry "Does logit masking change the model's relative preferences?".

**6.5 MED — schema-authoring pitfalls** (duplicate keys and missing required
keys are valid JSON; required keys need distinct FSM states).

**6.6 MED — self-refine/self-critique** (Huang et al. negative result: degrades
without external feedback; confirmation bias; works on style/grammar). qa entry
"why self-correction helps style but hurts math".

**6.7 MED — self-debugging beats self-consistency on sample count** (the "16
samples" figure unverified — do not import as fact).

**6.8 MED — library landscape** (llama.cpp GBNF is a CFG format; HF lacks
native structured output). **Landed** in L12b library list.

## 7. CMU 7+9 — chain-of-thought & reasoning models (`pKR3Vr6yg4U`, `6-mSbIPI4tc`)

**7.1 HIGH — reasoning length → decode cost is verified.** Each thinking token =
one 840 MiB re-read; a 10,000-token trace ≈ 8 TiB weight traffic, and the KV
grows to 10k × 112 KiB ≈ 1.1 GiB/step (> the weight read). **Landed** in L01
collapsible + qa entry + field-notes.

**7.2 HIGH — do NOT attribute "KV quantization breaks reasoning" to these
lectures** (they never mention KV compression). The mechanism is book-internal:
reasoning = long KV-dominated traces. **field-notes** updated to state the
mechanism, not a CMU claim.

**7.3 MED — test-time compute is bounded by context length** (sequential tokens
capped by max_seq_len; only *parallel* compute is free).

**7.4 MED — self-consistency "100× cost"** is the per-sequence number; batched,
the weight load is shared (the book's batch-K story).

**7.5 MED — distillation economics: `params × tokens`.** R1-32B can think ~21×
longer than 671B at equal decode cost. **Landed** in L28 + L28b lecture.

**7.6 LOW —** "470B parameters" is a factual error (R1 is 671B/37B); CoT helps
math/logic but not all tasks; GRPO advantage/clip verified correct.

## 8. CMU 10+11+12 — tools, agents, reward models (`prcsOlxc4vo`, `ixLXrgF77ME`, `p-MWR625HB8`)

**8.1 HIGH — scoring is prefill, generation is decode.** Scoring N candidates =
one compute-bound prefill; generating N = N memory-bound decodes. **Landed** in
qa entry + L24b.

**8.2 HIGH — agent loops drive long-context KV pressure.** 100–2000 steps =
10⁵–10⁷ tokens; the 8k crossover is crossed within ~100 steps. **Landed** in
qa.md long-context warning + L24b lecture.

**8.3 HIGH — best-of-N cost curve: ~20 → 32 SWE-bench for 16× inference.**
Linear cost, log-ish gain. **Landed** in L28b lecture + qa.

**8.4 MED — best-of-N KL bound `log n − (n−1)/n`** stated correctly and noted
non-tight (Beirami et al.). "Barami" is a garble of Beirami.

**8.5 MED — "non-deterministic even at T=0" (hosted APIs).** Kernel float
nondeterminism + model updates + prompt-template sensitivity; the oracle
guarantee holds for your own engine's argmax. **Landed** in qa entry.

**8.6 MED — prompt/KV caching is the agent-loop optimization**; "only the prompt
prefix is cached in practice."

**8.7 LOW — context condensation ~2× cost reduction** (unverified); CodeAct
(one big action vs many small tool calls) trades N prefills for one long
decode; MCP is the transport standard; reward models prefer longer outputs.

## 9. R1 / DeepSeekMath / Claude Code (`j4uj7hxsn5I`, `EteaXyBooHY`, `Y6uhwBF3F7k`, `pu311TSmwNQ`, `9-flnmRnU5I`, `jcNZ7fnD-Cg`)

Mostly training-side (record only). Inference imports:

**9.1 MED — reasoning "thinking" = decode cost + KV** (R1: answers stretch from
a couple lines to thousands of tokens; each is a decode step and a KV entry).
**Landed** in field-notes.

**9.2 MED — majority voting: ~64× cost for ~9 AIME points** (77.9 → 86.7).
**Landed** in field-notes + L28b.

**9.3 MED — agent context is the scarce resource; five-tier compaction ladder**
(trim → snip → micro-compact → collapse → auto-compact, cheapest first).
**Landed** in field-notes + L24b lecture.

**9.4 MED — tool definitions are prompt tokens; deferred schemas.** **Landed**
in L12b + L24b + field-notes.

**9.5 MED — subagents isolate context at ~7× token cost.** **Landed** in
field-notes + L24b.

**9.6 MED — harness-only change swings eval by ±18 points** (hold model fixed,
vary harness). **Landed** in field-notes.

**9.7 LOW —** R1 is 671B (not "600B"); std-dev in the GRPO example is 0.412 not
0.42; "schemes" = "skims".

## 10. DeepSeek-V3 MoE trio (`BtdCjsjmtNg`, `P9hHsrBoHEg`, `Vai47j7ovys`)

**10.1 HIGH — the 40% active-vs-dense example is absent.** DeepSeekMoE 16B/2.8B
does `2.8/7 = 40%` of a 7B dense LLaMA's arithmetic per token. **Landed** in L23.

**10.2 HIGH — the total-vs-active conflation inside the video.** "16B fits on a
40 GB GPU because only 2.8B is active" is wrong: it fits because total = 32 GB.
The book's warning is correct; used as a worked counterexample in L23.

**10.3 MED — MLA (multi-head latent attention) is a KV-compression technique the
book does not cover.** Cache a low-rank latent per token; rebuild K/V on the
fly. The V4 video describes sparse/summary siblings, not MLA itself, but the
gap is the whole *class* of structural KV compression. **Landed** in L05/L19 +
qa entry.

**10.4 MED — expert shape: V3 is 1 shared + 256 routed, top-8** (`8/256 =
3.1%`); 671/37 ≈ 18×. **Landed** in L23.

**10.5 MED — fine-grained segmentation + routing combinatorics**
(`C(16,2)=120 → C(64,8)≈4.4e9`); shared vs routed experts (dense vs sparse
serving profile). **Landed** in L23.

**10.6 MED — auxiliary-loss-free bias routing** (per-expert bias, no loss term,
never fights next-token prediction). **Landed** in L23 mitigations.

**10.7 MED — "skip tensor parallelism entirely" at 257 experts.** EP absorbs TP's
intra-node role; cross-node EP all-to-all is ~1:1 compute:communication and
needs DualPipe overlap. **Landed** in L23.

**10.8 MED — MTP = self-speculative decoding** (~85–90% acceptance, ~1.8
tokens/step). **Landed** in L12 Medusa bullet.

## 11. vLLM bite + V4 1M context + V3 bite (`OxYX8WPpypA`, `5mL7qxQ9GLw`, `aYIPTix80Pg`)

**11.1 HIGH — structural KV compression is entirely absent from L05/L19.** Sparse
attention, token-run summarization, low-rank latent compression — the second
lever beyond bytes. **Landed** in L05/L19.

**11.2 MED — BOOK ERROR, FIXED: the qa.md "~200k crossover" was garbled.** The
printed `~1e11/(300×2e5) ≈ 1.7 KB` evaluates to the *bytes-per-token input*, not
the crossover. Correct: `C = N_active × bytes/param / bytes_per_token` ≈
21–115M tokens for a frontier MoE — so 1M context stays *below* the crossover
once KV is compressed, which is exactly why V4's "10% of the KV cache" makes 1M
routine. **Landed** in qa.md (crossover entry rewritten).

**11.3 MED — V4 compression multiples** (27% of compute, 10% KV; flash model ~1/10
compute, 7% memory) — unverified against the paper; cited as claims.

**11.4 MED — MTP in the V3 bite** confirms §10.8.

**11.5 LOW — the book's "frontier MoE 256 experts, 32 active" was dated** (V3 is
8 active). Fixed to 18× / batch floor 5,400 in L28 + qa.

**11.6 LOW —** vLLM bite is a faithful summary (2–4×, 16-token blocks, COW all
match); "page detention" = PagedAttention, "copy on right" = copy-on-write,
"internal waste" ≠ the book's "internal fragmentation" (video's = reservation
waste); "utilization jumps from a third to nearly all" is the paper's ~30% →
~96% block utilization.

---

## 12. Book-side changes applied (the actionable set)

| # | Severity | Where | Change |
|---|----------|-------|--------|
| a | HIGH | L28, qa | Batch floor `B ≈ (F/BW)·(N_tot/N_act)`; ~76 dense-3090, ~5,400 MoE-FP4 (8/256 experts, 18×) |
| b | HIGH | L25, qa | ~25 ms per-step cadence floor (capacity/bandwidth) |
| c | HIGH | L01, qa, field-notes | Reasoning = test-time compute = decode traffic (10k trace ≈ 8 TiB) |
| d | HIGH | L06, qa | Greedy degeneration (coin-flip), why-not-beam, T=1 unbiasedness, epsilon/η/min-p family |
| e | HIGH | L03, qa | Greedy-determinism caveat (fp reduction order) |
| f | HIGH | L12b, qa | Token healing, FSM-vs-pushdown ceiling, deferred schemas, tool-defs-as-KV, GBNF |
| g | HIGH | L23, L05, L19, qa | DeepSeek MoE details + MLA/structural KV compression |
| h | MED | L22, L02, L12 | PP inference caveat; FP4 ≠ half of FP8; MTP self-speculation |
| i | MED | L28, L28b | Distillation economics (params × tokens); hosted-pricing signal |
| j | MED | L24b, field-notes | Serving agents: compaction ladder, subagents, best-of-N cost |
| k | LOW | glossary | Sampling/decoding section: search/model error, perplexity, beam, best-of-N |
| l | HIGH | qa.md | FIXED garbled crossover math (~200k → ~21–115M) and dated "32 active" → 8 |

## 13. New lectures added from this audit

- **L06b Search-based decoding** (`06b-search-based-decoding.md`, `engine/beam_search.py`): beam search, length normalization, A*/best-first, why sampling-with-a-verifier beat search.
- **L24b Serving agents** (`24b-serving-agents.md`, `engine/agent_context.py`): context as the scarce resource, compaction ladder, deferred schemas, subagents.
- **L28b Reasoning and test-time compute** (`28b-reasoning-and-test-time-compute.md`, `bench/reasoning_cost.py`): thinking = decode, majority-voting curves, context ceiling, distillation economics.

## 14. Verification appendix (key arithmetic re-derived)

- **Batch floor**: 3090 dense `F/BW = 76`, `N_tot/N_act = 1` → B ≈ 76. V3 `671/37 = 18.1`, FP4 `F/BW ≈ 300` → B ≈ 5,430.
- **Cadence**: 3090 24 GB / 936 GB/s = 25.6 ms; worst-case board+finish ≈ 2× ≈ 50 ms.
- **Reasoning**: 10,000 × 840 MiB = 8,400,000 MiB = 8.0 TiB; KV 10,000 × 112 KiB = 1.09 GiB.
- **Distillation**: 671/32 = 20.97 ≈ 21×; 671B × 1 token = 32B × 21 tokens.
- **Greedy degeneration**: 0.6¹⁰⁰ = 6.5e-23; 0.6⁶⁰·0.4⁴⁰ = 5.9e-30; C(100,60) = 1.37e28.
- **Crossover fix**: C = N_active×bytes/param / bytes_per_token = 37e9×1/1741 ≈ 21M (V3, fp8); book Qwen 840 MiB/112 KiB ≈ 7,500 ≈ 8k.
- **MoE**: 8/256 = 3.1%; 2.8/7 = 0.40; C(16,2)=120; C(64,8)=4,426,165,368.
- **Best-of-N**: 64× tokens ≈ 9 AIME points (77.9 → 86.7); KL bound log2−1/2 = 0.193 (n=2).
- **MTP**: 1 + 0.85 ≈ 1.85 ≈ 1.8 tokens/step ✓.
- **Subagent cost**: reported ~7× tokens (unverified, from Claude Code paper walkthrough).
- **Harness eval swing**: ±18 points (reported, secondhand, unverified).
- **LLM-as-judge / calibration**: post-RLHF logits are a ranking, not a confidence (literature; RLHF-degrades-calibration direction established, exact numbers unverified).

## 15. Value ranking (what mattered most)

1. **(l)** the qa.md crossover arithmetic error — a genuine book bug the V4 video exposed.
2. **(g)** MLA / structural KV compression — the largest *content* gap (whole class absent).
3. **(a)** the batch floor — L28's cost model had no batch dimension.
4. **(c)** reasoning = test-time compute — reframes serving cost for the current model generation.
5. **(d)** the sampling family + why-not-beam — L06's deepest conceptual gaps.
6. **(f)** token healing / expressiveness ceiling — L12b's two most practical misses.
7. **(j)** serving agents — new lecture built on the strongest applied material.

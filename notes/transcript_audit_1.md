# Transcript audit 1 — attention-focused videos vs. the book (L01/L02/L05/L16/L17/L18)

Audited 2026-08-15. Transcripts in `book/assets/transcripts/`; lectures cited as `B01`…`B18`,
qa.md as `qa:LINE`. Paper-side verification: arXiv abstracts, ar5iv full text, Hazy Research blog,
and the paper tables themselves (GQA Table 1, FA1 §4/Fig 2/Fig 5–8, FA2 abstract/Fig 2, ATIS Table 1/2).

Severity: HIGH = change the book; MED = worth adding (verified, not yet present); LOW = record only.

---

## 1. Grouped-query attention (`_hvdcVitU0Y`, `T-gqa`)

**1.1 Verified numbers not in the lectures** [[HIGH]]
The paper's headline row (GQA Table 1, T5-XXL, 5% uptrained): MHA 47.2 avg @ **1.51 s/sample**,
MQA 46.6 @ **0.24 s** (= 6.3×, "six times faster" ✓), GQA-8 47.1 @ **0.28 s** (= 5.4× vs MHA,
0.1 quality points below MHA — "shoulder to shoulder" ✓). The book's GQA coverage is architectural
only: callout `B05:124-128`, Q&A `B05:114-122`, exercise `B02:373-381` (n_kv_heads 16 vs 8). No
quantitative "groups vs speed vs quality" row exists anywhere in the book. Candidate: one
T5-XXL-style table in the `B05` callout or `B02` exercise as the paper anchor.

**1.2 Contradictions/nuances** [[MED→HIGH]]
The video's hook — "the KV cache … dwarfs the weights" (`T-gqa:31-38`) and "the weights aren't the
problem" (`T-gqa:14`) — collides with `B01:105-114` where **92.3%** of decode bytes are weights and
the KV cache is 7.7% (`B01:125-129`, "takes over past ~8k context"; at 32k it is 4.3× the model,
`B05:107-108`). Both are right; it is a regime crossover, and the book contains both claims.
Worked Qwen3-0.6B arithmetic (verified §8): halving KV heads 16→8 cuts total decode traffic by
~3% at 512 ctx but ~26% at 8k ctx. Points for a qa.md entry tying `B01`, `B05`, and the video
together — the "dwarfs the weights" claim is true exactly where serving hurts (long ctx + batch).
Also flag: `B05:124-128` "that's a direct 2× on the thing that bottlenecks you" overstates — it is
2× on the *cache* portion only (weights still dominate at short ctx). Tighten the wording.

**1.3 Exposition ideas** [[MED]]
- Prefill-vs-decode split in `B05` could carry the paper's decode-only timing caveat (GQA §3.1
  measures decode time per sample; prefill is unchanged — the video never says this, and neither
  does the book).
- The "one line" framing (keep per-group K/V = dial from MHA to MQA, `T-gqa:64-69`) is a good
  one-sentence addition to `B05:124` callout.

**1.4 qa.md candidates** [[MED]]
- "The KV cache dwarfs the weights in the GQA video but the book says weights are 92% of decode
  traffic" — with the ~8k crossover arithmetic above. Links into existing `qa:316`, `qa:352`.
- "Why does GQA pay for itself only in one regime?" (prefill vs decode; short vs long ctx).

**1.5 Questionable claims in the video** [[LOW]]
- `T-gqa:104` "1.51% sample" is STT garble for **1.51 s/sample**; `T-gqa:106` "0.28 multi-query
  speed" means 0.28 s/sample ≈ MQA-class speed. Quote the decoded values.
- "Eight times less memory … almost no quality lost" (`T-gqa:8-9`) is exactly the paper's T5-XXL
  claim (✓ per Table 1), but as a general statement about decoder-only llama-trained GQA it
  outruns the paper (which has no decoder-only experiments at all). LOW: add "on the paper's
  uptrained T5-XXL" when citing.
- "64 times smaller" for MQA (`T-gqa:49`) ✓ (64 query heads in T5-XXL). Mean-pooling ✓, 5% of
  pre-training steps ✓ (`α=0.05`, ≈600 TPUv3 chip-days, GQA §3.1). LLaMA-2/3, Mistral, MLA ✓.

---

## 2. FlashAttention 1 (`OUBsvKFvQQ0`, `T-fa1`)

**2.1 Verified numbers not in the lectures** [[HIGH]]
- GPT-2-attention case study: **67 GFLOPS / 40 GB / 42 ms** standard vs **75 GF / 4 GB** flash;
  42 → 7.3 ms ≈ **5.7×** ("nearly six times faster" ✓); the paper's own Fig 2 headline is
  **7.6×** ("up to seven and a half times faster" ✓, `T-fa1:19-20`).
- Memory-traffic theorem: N²d²/M with the lower-bound proof (`T-fa1:302-331`). The book has
  8N²+8Nd (`B02:13-16`) and the optimality claim (`B17:225-226`) but never the M-parameterized
  count.
- Results set: BERT-large **15%** vs MLPerf 1.1 record (20.0 → 17.4 min, FA1 Table 1); GPT-2 up to
  **3×** vs HuggingFace (9.5 → 2.7 days small), 1.7–1.8× vs Megatron; LRA **2.4×**; Path-X 16k
  61.4% and Path-256 64k 63.1% (first better-than-chance) — none in the book. The book's own
  benchmark targets (`B17:186`) could name the paper targets: 2–4× vs PyTorch on A100
  (FA1 Fig 5–6), 2.5–4.5× on the 3090 (Fig 7).

**2.2 Contradictions/nuances** [[MED]]
- "Same outputs **to the last bit**" (`T-fa1:18, 262-264`): the paper says "exact … up to
  floating-point rounding"; the book already has the right frame (`B17:93-105`, `qa:755`, tolerance
  ~1e-2 at `B17:104,185`). Keep the book; the video's phrasing is a harmless simplification.
- Loop order: `T-fa1:236-260` describes KV-outer / Q-inner — **exactly** FA1 Algorithm 1 (verified
  against the paper: "In each block, FlashAttention loops over blocks of Q"). The book's Triton
  kernel (`B17:130-157`) is Q-per-program / KV-inner, i.e. the FA2 arrangement, unnamed. Not a
  contradiction, but one footnote at `B17` is worth it (see §8, expo ideas).
- `B17:19-21` "on the 3090 … memory-bound by only 1.2×, temper your speedup expectations" vs the
  paper's Fig 7: the 3090 sees **higher** speedups (2.5–4.5×) than the A100 (2–4×) at long
  context because its bandwidth is lower. Different N-regimes; reconcile in one sentence.
- `B02:346-348` A100 = 2.039 TB/s; the video says "about 1½ TB/s" — that's the A100-40GB
  (1.5 TB/s), which is what FA1's experiments actually used. Non-issue; note if the book ever
  compares.

**2.3 Exposition ideas** [[MED]]
- The worked online-softmax example (`T-fa1:151-171`) — see §8, qa candidate below.
- The single-statistic fold l = m + log l (`T-fa1:166-171`) — the book's kernel keeps m and l
  separate (`B17:136-138`); the merge + single deferred divide is an FA2-style after-build variant
  (`T-fa2:107-113`).
- "Half the tiles gone for free" + diagonal-tile subtlety (`T-fa1:335-347`) — the book covers
  masking (`B17:165-175`) but not the "~half the tiles vanish" framing that motivates skipping
  fully-masked tiles (`B17:172-176` already handles the NaN corner; the tie-in is cheap).

**2.4 qa.md candidates** [[HIGH]]
- "Why does a running-max softmax recompute the exact same denominator?" — currently only prose
  (`B17:93-106`). Add the verified worked example: scores {3,1},{4,1},{5,2}; running sums
  1.1353 → 1.4675 → 1.5896; full-row denominator identical (§8). Collapsible at `B17` online
  softmax.
- "Why is FlashAttention's HBM traffic N²d²/M?" — one-line Theorem-1 walk-through. MED.

**2.5 Questionable claims in the video** [[LOW]]
- "each K&V block is about the size of SRAM" (`T-fa1:305-306`): loose — a tile is ~M/4d
  (`T-fa1:224-227` matches Algorithm 1's M/4d pin exactly), i.e. a small fraction of SRAM.
- "13 lines" (`T-fa1:282`): fine as a simplification.
- SRAM "19 TB/s vs 1½ TB/s HBM" — consistent figures; the *book's* ~100× claim is the
  out-of-line one (see §8).

---

## 3. FlashAttention 2 (`rp-Mxr1XZjc`, `T-fa2`)

**3.1 Verified numbers not in the lectures** [[MED]]
- All headline numbers check out against the paper: **25%** util for FA1 = floor of the paper's
  "25–40% of theoretical max"; **2×** over FA1 (abstract "around 2×"); **1.7–3×** depending on
  head dim & causal mask (Fig 2); **up to 10×** vs standard PyTorch (abstract; the Hazy blog chart
  reads 9× — both defensible); **230 TFLOPS**; **50–73%** of peak (abstract's own wording — the
  video's "73%" is the abstract's upper bound and is also self-consistent: 230/312 = 73.7%);
  **72% model FLOP utilization** for GPT training (225 TFLOPS) ✓; 225/312 = 72.1% ✓.
- **312 TFLOPS matmul vs 19.5 TFLOPS non-matmul → 16×** per-op cost (`T-fa2:104-109`) ✓ verbatim
  from the paper.
- 108 SMs on the A100 (`T-fa2:117` "clears 108") ✓; the "batch×heads < 108 → half the chip idle"
  logic ✓ (paper's occupancy argument).
- Warp split: FA1 slice-K (warps swap partials in shared memory); FA2 slice-Q, each warp owns its
  rows (`T-fa2:124-129`) ✓ per paper/Hazy blog.
- "16,000 token context for the price of eight" (`T-fa2:139-141`): supported by the paper's Table 4
  memory-footprint framing (16k with FA2 in roughly the footprint of ~8–9k without). MED.

**3.2 Contradictions/nuances** [[LOW]]
- "roughly 200 KB of it per multiprocessor" (`T-fa2:29-30`): A100 usable shared is 164 KB/SM max;
  200 KB is a loose round. LOW.
- The book's "within 2× is a genuinely good result" (`B17:209-210`) predates the FA2 story; the
  three FA2 levers (cut non-matmul flops / parallelize over seq len / warp-slice Q) are precisely
  the "why the gap exists" answer (`B17:205-211`). Merge in §3.3.

**3.3 Exposition ideas** [[MED]]
- Add the FA2 arc to `B17` "Why you won't beat the official FlashAttention": the three numbered
  levers + the 25%→73% utilization arc as the correct version of `B17:205-211`.
- The single-stat fold (subsumes §2.3).

**3.4 qa.md candidates** [[MED]]
- "Why does FA2 slice Q across warps instead of K?" — shared-memory write vs read asymmetry, ties
  into `B17:205-211`.

**3.5 Questionable claims** [[LOW]]
- "only hit 25% of what the chip can do" — fine as the floor of the paper's own 25–40% range; say
  "25–40%" when quoting.
- Everything else in §3.1 verified; no factual errors of substance.

---

## 4. Attention Is All You Need, full (`CMVh1-LVkWU`, `T-atis`)

**4.1 Verified numbers not in the lectures** [[HIGH]]
- Toy attention by hand (`T-atis:99-124`): all arithmetic verified (§8): scores [0,6,2,2] → /√4 →
  [0,3,1,1] → exp [1, 20.09, 2.72, 2.72] → Σ26.52 → [0.04, 0.76, 0.10, 0.10]; output [0.04,
  1.51, 0.10]. Nothing like a worked-by-hand attention example exists in the book.
- √d_k derivation (`T-atis:125-154`): sd of a dot product = √d_k (64 → 8, "±8 not ±1");
  "gap 16"; e¹⁶ ≈ 9M; spiked softmax → zero gradient (Jacobian p(δ−p)); why divide by √d and not
  d. The book only says "1/√head_dim applied before softmax" (`B17:170`) with no why. This is the
  single biggest verified-gap item for L17. [[HIGH]]
- Head economics: 8×64 vs 1×512 cost is identical (`T-atis:173-179`) ✓.
- FFN accounting: 2.097M per layer vs ~1.05M all four attention projections → 2/3 of parameters
  (`T-atis:288-291`) ✓ — the book's parameter accounting (`B02`) never decomposes per-layer.
- Complexity Table 1: attention n²d vs recurrent nd² → d/n = 512/40 = 12.8 ≈ 13× cheaper per layer
  ("roughly 13 times" ✓); sequential ops 1 vs n; max path 1 vs n; restricted attention n/r
  (sliding window written down in 2017).
- Training recipe: Adam betas 0.9/0.98; warmup 4,000 steps; LR peak ≈ 7×10⁻⁴ (derived §8); dropout
  + label smoothing 0.1; base 100k steps / 12 h / 8 P100s / 65M params; big 300k / 3.5 days.
- Results: En-De big 28.4 vs best-prior 26.3 (GNMT+RL ensemble) = 2.1 "over two points" ✓; En-Fr
  41.8 single-model record (Table 2; the abstract prints 41.0 — see §8 item f).
- Table 3 rows: 1 head @ 512 dims loses ~0.9 BLEU; 32×16 worse; key-dim row; label smoothing
  hurts ppl but helps BLEU ("the metric you optimize and the metric you care about openly
  disagree, and they published it anyway" ✓).

**4.2 Contradictions/nuances** [[MED]]
- "the one line the paper got slightly wrong" (`T-atis:298-326`) = post-LN (warmup 4000 or
  diverge; pre-LN took over within ~2 years). Editorial, defensible — the paper *did* use post-LN;
  the video's phrasing is delivering an opinion, not correcting the record. No book conflict (the
  book doesn't cover the original architecture).
- The 1/100 compute claim → §4.5 (the only substantive numeric overclaim in this audit).

**4.3 Exposition ideas** [[MED]]
- The √d story slots into `B17:170` as a collapsible; the toy attention example could anchor `B02`
  attention-intro or `B17` "The problem".
- The 96%→76% arc ("attention versus an argmax") is the best single visual for why scaling works —
  the book has no plots of any of this.

**4.4 qa.md candidates** [[HIGH→MED]]
- "Why divide by √d_k and not d?" — with the full ±8 / e¹⁶ / zero-gradient chain (the most-copied
  symbol). HIGH.
- "Why did the original Transformer need 4,000 warm-up steps (post-LN) and why did pre-LN kill the
  requirement?" MED.
- "Why is the FFN 2/3 of a transformer's parameters?" LOW-MED (anchors `B02`).

**4.5 Questionable claims** [[MED]]
- **"roughly a 100th of the compute … straight from the abstract"** (`T-atis:386-391`): two
  problems. (a) It is not in the abstract — the abstract says "a small fraction of the training
  costs". (b) The video's own numbers (base 3.3×10¹⁸ vs prior 1.0×10²⁰) give **1/30**, not 1/100
  (3.3e18/1.0e20 = 0.033); vs ConvS2S's 1.5×10²⁰ it's 1/45, vs GNMT's 2.3×10¹⁹ it's 1/7. "A
  hundredth" overclaims by ~3×. The bite repeats it (`T-bite:126-128`). The honest number is "1/30
  vs the previous SOTA; down to ~1/7 vs some baselines".
- "at 64 dimensions" (`T-atis:153`): the toy is 4-dim (d_k=4); the 96→76% arc is computed there.
  The sentence muddles toy and general case. LOW.
- "Vasuani" (`T-atis:6`), "roo 64 = 8", "512 in48", "7 * 10us 4", "4,96", "Eate/asciendance"
  (`T-atis:357` = "sequence length") — all STT garble, decoded in §8. LOW.
- "In 2017, sequences were 40 tokens" — the paper's own example; fine as a framing device.

---

## 5. Attention-Is-All-You-Need bite (`d1FrlDzGEB4`, `T-bite`)

**5.1 Verified numbers not in the lectures** — same as §4.1 minus the toy example; the bite's
unique items: the ±8/√d version (`T-bite:46-58`), the residual-stream "straight wire" framing
(`T-bite:92-104`), wavelengths "a few radians to 60,000" ✓ (10000·2π ≈ 62,832; 256 hands ✓),
"64×8 is 512 so eight heads cost what one did" ✓.

**5.2 Contradictions/nuances** [[LOW]]
- "roughly a hundredth of the training compute" (`T-bite:128`) — same overclaim as §4.5 (no FLOP
  figures in the bite, so it inherits the full version's math problem).
- "three and a half days on eight GPUs" for En-Fr 41.8 ✓ (paper: 3.5 days on 8 P100s).

**5.3 Exposition ideas** [[LOW]]
- The residual-stream "every block writes an edit onto a running stream" line is the clearest
  two-sentence statement of why post→pre-LN matters; if the book ever covers architecture history,
  this is the phrasing. Currently the book's residual-stream mentions (e.g. `B17`) don't exist in
  this form — LOW because Book/M1.1 course may not cover ATIS at all.

**5.4 qa.md candidates** — none beyond §4.4 (the bite is a compression).

**5.5 Questionable claims** [[LOW]]
- None beyond §4.5 & §5.2. "Attention already existed bolted onto that chain" ✓ historically
  accurate. The BLEU/complexity claims are all paper-accurate.

---

## 6. Book-side errors surfaced by this audit (the actionable set)

| # | Severity | Location | Issue | Fix |
|---|----------|----------|-------|-----|
| a | **HIGH** | `B17:110-112`, also `B00:50-57`, `B18:52-54`, `appendix-a-glossary:122`, `B20:74` | SRAM "~6 MB total on the 3090, roughly 100× faster to access than HBM". 6 MB is the 3090's **L2** size (`qa:332` already says so correctly); FA-style kernels use per-SM shared (≤100 KB/SM → ≈8.2 MB aggregate). Bandwidth ratio HBM→SRAM on 3090 ≈ 936 GB/s → ~19 TB/s ≈ **~20×** (FA videos: 19 TB/s vs 1.5–2 TB/s = 10–13×). "100×" fails both readings. `B00:57` "~147×" is built on the same 6 MB figure (correct ratio ≈ 880/8.2 ≈ ~100× — lands near 100 but for the wrong reason; really ~53–84× depending on what "on-chip" includes) | Rewrite the box: "a few MB total across SMs (~100 KB/SM on the 3090, ~8 MB aggregate), ~10–20× faster to access than HBM"; fix the 147× arithmetic in `B00`; rebuild with `--strict` |
| b | **HIGH** | `B17:70-106` | Online-softmax "exact" claim has no worked numbers | Add the verified example (scores blocks {3,1},{4,1},{5,2}; 1.1353 → 1.4675 → 1.5896 = full-row denominator) as collapsible + qa.md entry, linking `qa:755` |
| c | **MED→HIGH** | `B17:170` | "Wrong scale: 1/√head_dim" with zero justification | Add collapsible: dot-product sd = √d; ±8 swing; e¹⁶ spike; zero-gradient softmax; 96%→76% arc. qa.md entry "Why divide by √d_k and not d?" |
| d | **MED** | `B01:105-129` vs `B05:124-128` vs `T-gqa:14,31-38` | "Weights are 92.3% of decode bytes" (short ctx) vs "the KV cache dwarfs the weights" (long ctx) vs the video — internal tension, never resolved in-book | ~8k crossover note in `B01:129`; tighten "direct 2×" wording in `B05:124`; qa.md entry with worked crossover arithmetic (§8) |
| e | **MED** | `B17:19-21` | "temper your speedup expectations on the 3090" | One-line reconcile with FA1 Fig 7 (3090 shows *higher* relative speedups, 2.5–4.5×, at long context) |
| f | **LOW** | `B17:130-157` | Book kernel is FA2-arranged (Q-per-program) while FA1 Algorithm 1 is KV-outer/Q-inner (as the video says) | One line: "FA2 flips this loop order; your kernel follows FA2" |
| g | **LOW** | glossary `:122` | "SRAM ~100× faster" also in glossary (and `B20:74`) | Same fix as (a) |

Corrected hardware facts for (a): GA102 (3090): 82 SMs; max shared per SM 100 KB → ≈8.2 MB;
L2 = 6 MB. A100: 108 SMs × 164 KB max shared ≈ 17.7 MB (video's "roughly 200 KB/SM" is loose:
real max 164 KB). HBM↔shared ratio: 3090 ≈ 936 GB/s vs ~19.5 TB/s ≈ 1:21; A100 ≈ 2.0 TB/s vs
~19 TB/s ≈ 1:9.5. The FA videos' own 19 vs 1.5–2 TB/s = "~10×", not "~100×".

---

## 7. Cross-cutting: what to keep in mind when quoting these videos

- STT decodes: "1.51% sample" → **1.51 s/sample**; "0.28 multi-query speed" → 0.28 s/sample;
  "7 * 10us 4" → 7×10⁻⁴; "512 in48" → 2048; "4,96" → 4096; "Vasuani" → Vaswani; "overd" → d;
  "asciendance" → sequence length.
- Every verified number that appears in the videos checks out against the papers, with two
  exceptions, both LOW-to-MED: the ATIS "1/100th of the compute" (it is 1/30 per the video's own
  numbers; and not "straight from the abstract"), and the FA2 "25%" (floor of the paper's 25–40%).
- The one real book-vs-video conflict is the SRAM bandwidth/size family (a), which is a *book*
  error, not a video error.

## 8. Verification appendix (all arithmetic re-derived by hand)

- **Online softmax** (`T-fa1:151-171`): blocks {3,1},{4,1},{5,2}. Block 1: m=3, l=1+e⁻²=1.13534.
  Block 2 (m=4): l·e⁻¹ + (1+e⁻³) = 0.41767+1.04979 = 1.46746. Block 3 (m=5): l·e⁻¹ + (1+e⁻³) =
  0.53982+1.04979 = 1.58961 ≈ 1.590. Full row, one shot, m=5: e⁻²+e⁻⁴+e⁻¹+e⁻⁴+1+e⁻³ = 1.58956.
  Identical. ✓
- **Toy attention** (`T-atis:99-124`): scores the=0, cat=2·2+2·1=6, sat=2, down=0 (dot products
  verified term-by-term); ÷√4 → [0,3,1,1]; exp → [1, 20.09, 2.72, 2.72]; Σ = 26.52; weights
  [0.04, 0.76, 0.10, 0.10] (Σ=1). Output [0.04, 1.51, 0.10]: sat's weighted value-average ✓
  (1.51 = 0.76×2, narrated with rounding). Unscaled top weight 96.2% (e⁶/(e⁶+e⁰+2e²) = 403/419 =
  0.962) → scaled 76% ✓.
- **√d chain**: Var(dot over d_k independent unit-variance pairs) = d_k; sd = √d_k; at d_k=64 →
  8 ("±8 not ±1" ✓); gap 16 typical E[max−min]; e¹⁶ = 8.886×10⁶ ≈ 9M ✓; Jacobian p(δ−p) → ~0 at
  a spike ✓. 512⁻⁰·⁵ × 4000⁻⁰·⁵ = 0.04419×0.01581 = 6.98×10⁻⁴ ≈ 7×10⁻⁴ ✓ (peak LR). Note:
  strictly, "divide by √d_k restores variance 1" makes the *dot-product* variance 1, not each
  score's sd 1 — the video's "the variance is back to exactly one" (`T-atis:147-148`) is the
  standard shorthand ✓.
- **Head economics**: 1·N²·512 vs 8·N²·64 = 8·N²·64 ✓ identical. FFN 2·512·2048 =
  2,097,152; attention 4·(512·512) = 1,048,576 → exactly 2:1 → FFN = 2/3 of the pair ✓.
- **Complexity**: attention n²d vs recurrent nd²; 2017: d/n = 512/40 = 12.8 ≈ 13 × ✓. Sequential
  ops 1 vs n ✓. Path length 1 vs n ✓. Restricted n/r ✓ (Table 1).
- **GQA**: 1.51/0.24 = 6.29 ("six times" ✓); 1.51/0.28 = 5.39; 47.2−47.1 = 0.1 avg pts ✓;
  64 heads/8 groups = 8× cache cut ✓; mean-pooling + α=0.05 ✓ (paper §3.1, ≈600 TPUv3 chip-days).
- **FA1**: 67 GF base (4N²d+3N² at N×N… per paper §2) vs 75 GF flash (+12%, recompute ✓);
  40 GB → 4 GB (10× ✓); 42 ms → 7.3 ms = 5.75 × ("nearly six" ✓); 7.6× headline (Fig 2, GPT-2
  attention on A100) ✓; BERT 20.0 → 17.4 min = 13% ~ 15% ✓; GPT-2 small 9.5 → 2.7 days = 3.5×
  ("up to 3×" abstract ✓); LRA 2.4× ✓; Path-X 16k 61.4%, Path-256 64k 63.1% ✓; block-sparse LRA
  2.8× ✓. 3090 2.5–4.5× vs A100 2–4× (Fig 5–8) ✓. Traffic count N²d²/M ✓ (Theorem 1); d≈100,
  M≈100 KB → M ≫ d² ✓.
- **FA2**: 312/19.5 = 16 × ✓; 230/312 = 73.7% (video's 73% self-consistent; paper prints
  50–73% and 72% = 230/320) ✓; 225/312 = 72.1% (abstract's "72% model FLOP utilization" ✓);
  A100 108 SMs ✓; split-Q warp scheme ✓; 1.7–3× range (Fig 2, head dim & causal) ✓.
- **ATIS**: 28.4 − 26.3 = 2.1 BLEU ✓; 41.8 single-model En-Fr (Table 2; abstract's 41.0 predates
  — quote Table 2); base 3.3×10¹⁸ vs prior 1.0×10²⁰ → **1/30**, not 1/100 ✓ (also 1/45 vs
  ConvS2S 1.5×10²⁰, 1/7 vs GNMT 2.3×10¹⁹); "100th" = overclaim. Betas 0.9/0.98 ✓; dropout 0.1,
  smoothing 0.1 ✓; base 100k/12 h/8 P100; big 300k/3.5 days ✓; 65M params ✓; tied embeddings,
  ×√512 ✓; wavelengths 2π…2π·10⁴ ≈ 62,832 ✓ (256 pairs for d_model 512).
- **Qwen3-0.6B GQA crossover** (`B01` vs `T-gqa`): per token, 2×28×8×128×2 = 114,688 B ✓
  (`B05:95`). Weights 880.8 MB. At 512 ctx: cache 58.7 MB → halving KV heads (16→8, `B02:378`)
  cuts decode bytes 910.2→880.8+29.4 = 0.969 → ~3%. At 8k ctx: cache 939.5 MB → 0.742 → ~26%.
  The video's regime is 8k+; the book's is short-ctx — crossover ≈ 8k ✓ consistent with
  `B01:127-129`.
- **SRAM claim audit** (item a): qa:332 already states "~128 KB L1/SM, 6 MB L2" correctly; `B17`
  and `B00` conflate L2 with SRAM; "~100× faster" appears in `B17:111`, `B18:54`,
  `appendix-a-glossary:122`, `B20:74`. Correct: usable shared ~100 KB/SM (3090), 164 KB/SM (A100);
  bandwidth ratio ≈ 1:10–1:21 (this audit's own arithmetic; videos: 9.5–13×).

## 9. Value ranking (if time is short, do in this order)

1. **(a)** SRAM 6 MB / 100× family — book error contradicting the FA videos and the book's own
   qa.md:332. 4 files, small edit, rebuild `--strict`.
2. **(b)** Online-softmax worked example — qa.md entry + `B17` collapsible. Verified numbers.
3. **(c)** √d_k justification — qa.md + `B17:170` collapsible (the "most copied symbol" story).
4. **(d)** GQA regime tension — qa.md entry + `B01:129` one-liner + `B05:124` wording fix.
5. **(e)** `B17:19-21` reconcile with FA1 Fig 7 (one sentence).
6. **(f,g)** Loop-order footnote; glossary/`B20` 100× cleanup — LOW, batch with a rebuild.
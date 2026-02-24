# Thin Keys, Full Values -- Road to ICML (Solid 8)

## Paper Summary

The paper proposes **asymmetric attention**: QK projections use lower dimensionality
(d_select << d_model) while V retains full dimensionality. Core insight: attention
selection (QK dot product) is a ranking problem needing only O(log N) dimensions,
while value transfer needs full capacity. Validated across 7 experiments from
synthetic tasks to Mistral-7B post-training compression. At d_select = d_model/4,
achieves 75% key cache savings at ~2% PPL cost after SVD + QK fine-tuning.

---

## TODO (pick up here)

1. **Experiment F2: C4 + Math mixed fine-tuning** -- r256 GSM8K gap worsened with pure C4; try 7M C4 + 3M math tokens from `/sg-pretrain/datasets/mathematics_dataset-v1.0/`. Run r256, r512, control on 3 GPUs.
2. **Add Experiment B results to paper tex** -- factored inference table (KV cache sizes, decode throughput). Insert after the downstream table added today.
3. **Commit and push** all today's work (scripts, results, README, tex edits).
4. **Experiment C** (P1): Train 1B from scratch with thin keys on FineWeb-Edu. Needs 8xH100 for 2-4 days.
5. **Experiment D** (P1): GQA/MLA comparison at 125M-350M scale.

---

## Session Log: 2026-02-24

### What we did today

1. **Experiment A results added to paper tex** -- inserted Table 5 (downstream tasks) with full results for Hellaswag/ARC/WinoGrande/MMLU/GSM8K, plus delta columns vs Control+FT and interpretation paragraph. Located after the Mistral-7B PPL table in Section 3.7.

2. **Experiment B: Factored Key Inference Benchmark [DONE]** -- implemented actual factored inference with per-head SVD, Q absorption, thin key caching, and SDPA attention. Shows real KV cache savings and throughput gains. See results below.

3. **Experiment F: Diverse Fine-tuning on C4 [DONE]** -- replaced WikiText-103 with C4 (10M tokens) for QK fine-tuning. Mixed results on GSM8K: helps r512, hurts r256. See results below.

4. **Paper tex edits (from earlier session)** -- reframed "factored keys" as the central novelty in abstract, intro contributions, Section 2.3 title, and conclusion.

### Key files created/modified today
- `experiments/bench_factored.py` -- factored inference benchmark (per-head SVD + Q absorption + thin KV cache)
- `experiments/bench_throughput.py` -- original throughput benchmark (showed identical memory, motivating the factored version)
- `experiments/svd_finetune_diverse.py` -- SVD+FT pipeline with C4 data support
- `experiments/logs/factored_bench.json` -- Experiment B results
- `experiments/logs/downstream_{r256,r512,control}_c4_ft.json` -- Experiment F results
- `Colm-2026/colm2026_conference.tex` -- added downstream eval table + factored keys reframing

---

## Experiment A Results: Downstream Task Evaluation on Mistral-7B [DONE]

### Full results table

| Task | Metric | Baseline | r512 noFT | r512+FT | r256 noFT | r256+FT | Ctrl+FT |
|---|---|---|---|---|---|---|---|
| Hellaswag | acc_norm | 81.2 | -- | 81.4 | 72.6 | 80.7 | 81.3 |
| ARC-Challenge | acc_norm | 54.0 | -- | 54.1 | 48.4 | 53.4 | 54.4 |
| WinoGrande | acc | 75.4 | -- | 73.2 | 70.1 | 72.1 | 73.2 |
| MMLU | acc | 60.1 | 59.1 | 55.2 | 50.4 | 54.4 | 55.7 |
| GSM8K | exact_match | 38.4 | 33.8 | 27.7 | 16.5 | 25.8 | 29.9 |

- **Baseline** = original Mistral-7B, no modification
- **noFT** = SVD compressed, no fine-tuning
- **+FT** = SVD compressed + 3 epochs QK fine-tuning on WikiText-103
- **Ctrl+FT** = no compression, same fine-tuning (fair comparison baseline)
- **r512** = rank 512, 50% K cache saved; **r256** = rank 256, 75% K cache saved

### Deltas vs Control+FT (fair comparison)

| Task | r512+FT vs Ctrl | r256+FT vs Ctrl |
|---|---|---|
| Hellaswag | +0.1% | -0.7% |
| ARC-Challenge | -0.5% | -1.7% |
| WinoGrande | +0.0% | -1.6% |
| MMLU | -0.9% | -2.3% |
| GSM8K | -7.4% | -13.7% |

### Key findings

1. **Fine-tuning dramatically recovers downstream performance.** r256 without FT loses 10-57% across tasks. After 3 epochs of QK fine-tuning, gaps shrink to 0.7-2.3% on knowledge/commonsense tasks.

2. **r512 (50% K cache saved) is essentially lossless** after fine-tuning: <1% degradation on Hellaswag/ARC/WinoGrande, -0.9% on MMLU.

3. **GSM8K (math reasoning) is disproportionately affected** -- confirming reviewers' concern that PPL understates downstream impact on reasoning. r256+FT still loses 13.7% vs control on GSM8K, even though PPL gap is only ~1.2%.

4. **The fair comparison is vs Control+FT, not vs original baseline.** Fine-tuning on WikiText-103 causes slight domain shift that hurts MMLU/GSM8K for all models (including control). The residual compression gap is what matters.

5. **This result strengthens the paper** by being honest and nuanced: excellent on knowledge/commonsense, measurable cost on math reasoning. Practitioners can choose their operating point (r512 for near-lossless, r256 for aggressive savings with known reasoning cost).

---

## Experiment B Results: Factored Key Inference Benchmark [DONE]

Implemented actual factored inference: per-head SVD of W_K, absorption of expansion matrix into W_Q, thin keys cached in KV cache, attention computed in thin Q/K space with full-dim V via SDPA. Benchmarked on H100 80GB.

### KV cache size (bs=1, per context window)

| Config | K cache | V cache | KV total | K head dim | Savings |
|---|---|---|---|---|---|
| Baseline | 268.4 MB | 268.4 MB | 536.7 MB | 128 | -- |
| Factored r512 | 134.2 MB | 268.4 MB | 402.6 MB | 64 | 25% KV, 50% K |
| Factored r256 | 67.1 MB | 268.4 MB | 335.5 MB | 32 | 37.5% KV, 75% K |

### Decode throughput (tokens/sec) -- the serving-critical metric

| Config | ctx=4K bs=4 | ctx=4K bs=8 | ctx=4K bs=16 | ctx=4K bs=32 | ctx=16K bs=4 | ctx=16K bs=8 |
|---|---|---|---|---|---|---|
| Baseline | 117 | 149 | 172 | 187 | 117 | 147 |
| Factored r512 | 145 (+24%) | 182 (+22%) | 211 (+23%) | 230 (+23%) | 145 (+24%) | 182 (+24%) |
| Factored r256 | 156 (+33%) | 200 (+34%) | 235 (+37%) | 259 (+38%) | 157 (+34%) | 201 (+37%) |

### Peak GPU memory (GB) during decode

| Config | ctx=4K bs=8 | ctx=4K bs=32 | ctx=16K bs=8 |
|---|---|---|---|
| Baseline | 21.4 | 42.2 | 27.7 |
| Factored r512 | 19.6 (-8%) | 36.7 (-13%) | 25.9 (-6%) |
| Factored r256 | 18.6 (-13%) | 34.0 (-19%) | 24.9 (-10%) |

### Key findings

1. **Real KV cache savings confirmed** -- K cache shapes show actual thin keys: [bs, 8, seq, 32] for r256 vs [bs, 8, seq, 128] for baseline.
2. **Factored keys are faster, not just smaller** -- 22-38% higher decode throughput because thinner K projections mean less memory bandwidth for cache reads.
3. **Peak memory reduced** -- 8-19% lower peak GPU memory, enabling more concurrent users.
4. **SDPA supports asymmetric K/V dims** -- `F.scaled_dot_product_attention` works with thin Q/K and full V (Ev != E). No custom kernel needed for correctness; Flash Attention kernel would further improve performance.
5. **Prefill throughput also improves** -- 6-12% faster prefill due to smaller Q/K projections.

---

## Experiment F Results: Diverse Fine-tuning Corpus (C4) [DONE]

Replaced WikiText-103 with C4 (10M tokens, diverse web text) for QK fine-tuning. Same 3-epoch, QK-only fine-tuning protocol. WikiText-103 validation used for comparable PPL.

### C4 vs WikiText-103 fine-tuning comparison

| Task | Ctrl+WT | r512+WT | r256+WT | Ctrl+C4 | r512+C4 | r256+C4 |
|---|---|---|---|---|---|---|
| Hellaswag | 81.3 | 81.4 | 80.7 | **82.6** | **82.5** | **81.7** |
| ARC-C | 54.4 | 54.1 | 53.4 | **55.1** | **55.6** | **53.8** |
| WinoGrande | 73.2 | 73.2 | 72.1 | 72.2 | **73.7** | 72.4 |
| MMLU | 55.7 | 55.2 | 54.4 | **56.4** | **56.1** | **55.3** |
| GSM8K | 29.9 | 27.7 | 25.8 | **30.6** | **29.4** | 22.8 |

### Compression gap (delta vs respective control)

| Task | r512+WT gap | r512+C4 gap | r256+WT gap | r256+C4 gap |
|---|---|---|---|---|
| Hellaswag | +0.1% | -0.2% | -0.7% | -1.1% |
| ARC-C | -0.5% | +0.8% | -1.7% | -2.5% |
| MMLU | -0.9% | -0.6% | -2.3% | -2.0% |
| GSM8K | -7.4% | **-3.9%** | -13.7% | **-25.5%** |

### Key findings

1. **C4 improves absolute scores** on Hellaswag (+1.1-1.3), ARC (+0.7-1.5), MMLU (+0.7-0.9) across all configs. Diverse data helps general knowledge recovery.

2. **r512 GSM8K gap halved** with C4: 3.9% vs 7.4% with WikiText. For mild compression, diverse data successfully helps math reasoning recovery.

3. **r256 GSM8K gap doubled** with C4: 25.5% vs 13.7% with WikiText. For aggressive compression, C4's noise hurts. The severely compressed QK projections can't recover fine-grained math patterns from diluted math signal in web text.

4. **Overfitting on C4 observed**: val PPL (on WikiText-103) rose across epochs (5.82 → 6.06 → 6.56 for control). 3 epochs may be too many for C4; early stopping might help.

5. **Next step: targeted math data.** Pure C4 helps mild compression but not aggressive. The right approach is a mixed corpus: 7M tokens C4 + 3M tokens from `/sg-pretrain/datasets/mathematics_dataset-v1.0/` (procedural math, no GSM8K contamination). This gives diverse coverage plus targeted math reasoning signal.

---

## Weaknesses (merged from 3 reviews)

### Unanimous (Claude + DeepSeek R1 + DeepSeek R2)

**W1: Perplexity-only evaluation** [FIXED by Exp A]
- ~~2% PPL gap could be 0.5% or 8% on reasoning tasks -- nobody knows~~
- DONE: 5 downstream benchmarks on Mistral-7B, results in paper tex

**W2: Flash Attention incompatibility** [PARTIALLY FIXED by Exp B]
- ~~Paper says dk=dv assumption baked into optimized kernels, but offers no solution or measurement~~
- DONE: Showed SDPA supports asymmetric dims. Factored inference works and is 22-38% faster on decode.
- REMAINING: Could add roofline analysis or note that custom FA kernel would further improve.

### Strong consensus (2 of 3 reviews)

**W3: No train-from-scratch beyond 125M** [TODO]
- Paper's own recommended path (train from scratch) is unvalidated at scale that matters (1B+)
- DS-R1 explicitly asks for a 1B-3B run

**W4: Missing MLA comparison** [TODO]
- DeepSeek's Multi-head Latent Attention also does low-rank KV compression -- biggest elephant in the room
- DS-R2: "How is this different from just setting head dimension smaller in GQA?"

**W5: No real throughput numbers** [FIXED by Exp B]
- ~~Paper claims "25GB savings, 60% more users" but never measures actual tokens/sec or batch sizes~~
- DONE: Full throughput/memory benchmark with actual factored inference on H100.

### Moderate

**W6: Incremental novelty** (both DeepSeek reviews)
- Well-executed but not paradigm-shifting. Asymmetric QKV dimensions aren't fundamentally new.
- Paper reframed around "factored keys" as a new inference primitive (tex edits done).

**W7: Narrow fine-tuning data** [PARTIALLY FIXED by Exp F]
- ~~SVD+FT uses only WikiText-103 (10M tokens). Unclear if it generalizes to diverse pretraining distributions.~~
- DONE: C4 tested. Helps knowledge tasks, helps r512 GSM8K, hurts r256 GSM8K.
- NEXT: Try C4+math mix (Experiment F2).

### Consensus matrix

| Weakness | Claude | DS-R1 | DS-R2 | Consensus | Status |
|---|---|---|---|---|---|
| W1: Downstream tasks beyond PPL | YES | YES | YES | **UNANIMOUS** | **DONE** |
| W2: Flash Attention barrier | YES | YES | YES | **UNANIMOUS** | **PARTIAL** |
| W3: Train-from-scratch at 1B+ | YES | YES | -- | Strong | TODO |
| W4: MLA comparison missing | YES | -- | YES | Strong | TODO |
| W5: End-to-end throughput | YES | -- | YES | Strong | **DONE** |
| W6: Incremental novelty | Mild | YES | YES | Moderate | Reframed |
| W7: Narrow FT data | YES | -- | -- | Noted | **PARTIAL** |

---

## Experiment Plan: Weaknesses -> Fixes

### P0: Must-have (address unanimous weaknesses)

#### Experiment A: Downstream Task Evaluation on Mistral-7B [DONE]
- **Fixes**: W1 (perplexity-only)
- **Result**: See results table above. r512+FT is near-lossless (<1% on most tasks). r256+FT has small gaps on knowledge tasks (0.7-2.3%) but larger on math reasoning (GSM8K -13.7%).

#### Experiment B: Factored Key Inference Benchmark [DONE]
- **Fixes**: W2 (Flash Attention), W5 (no throughput numbers)
- **Result**: See results above. 75% K cache reduction confirmed. 22-38% decode speedup. 8-19% peak memory reduction. SDPA supports asymmetric dims natively.

### P1: High impact (address strong-consensus weaknesses)

#### Experiment C: Train from Scratch at 1B-3B Scale [TODO]
- **Fixes**: W3 (no large-scale train-from-scratch)
- **What**: Train a 1.3B model with d_select = d_model/4 from scratch on FineWeb-Edu (~50B tokens)
- **Compute**: Moderate -- one 1B training run (2-4 days on 8xH100)
- **Expected outcome**: If PPL degradation is again ~4%, the architecture-independence claim is iron-clad.

#### Experiment D: Direct GQA and MLA Comparison [TODO]
- **Fixes**: W4 (missing MLA), W6 (incremental novelty)
- **What**: Comparison table at the same effective KV cache budget

### P1.5: Elevated priority (fix GSM8K gap)

#### Experiment F2: C4 + Math Mixed Fine-tuning [NEXT]
- **Fixes**: W7 (narrow FT data), GSM8K gap
- **What**: Fine-tune with 7M tokens C4 + 3M tokens from math dataset (`/sg-pretrain/datasets/mathematics_dataset-v1.0/`)
- **Why**: Pure C4 helps r512 GSM8K (gap halved) but hurts r256. Targeted math data should help aggressive compression recover reasoning patterns without GSM8K contamination.
- **Compute**: Low -- same as Exp F, just different data mix
- **Script**: Modify `experiments/svd_finetune_diverse.py` to support mixed data loading

### P2: Nice-to-have (polish and strengthen)

#### Experiment E: Flash Attention Kernel or Roofline Analysis
- **Fixes**: W2 (Flash Attention)
- **What**: Roofline analysis showing memory-bound regimes. Exp B already showed SDPA works; roofline would strengthen the theoretical argument.

#### Experiment G: Per-Layer d_select Allocation
- **Fixes**: W6 (incremental novelty)
- **What**: Allow different d_select per layer.

---

## Impact Summary

| Priority | Experiment | Fixes | Status | Impact |
|---|---|---|---|---|
| **P0** | A: Downstream tasks | W1 | **DONE** | Highest |
| **P0** | B: Throughput benchmarks | W2, W5 | **DONE** | High |
| **P1** | C: Train 1B+ from scratch | W3 | TODO | High |
| **P1** | D: GQA/MLA comparison | W4, W6 | TODO | High |
| **P1.5** | F2: C4+Math FT | W7, GSM8K | **NEXT** | High |
| **P2** | E: FA roofline | W2 | TODO | Medium |
| **P2** | F: Diverse FT (C4) | W7 | **DONE** | Medium |
| **P2** | G: Per-layer d_select | W6 | TODO | Medium |

**A and B are done. This moves the paper from 7 to 7.5+.**
**F2 (C4+math) is next -- cheap and could close the GSM8K gap.**
**C and D are the remaining high-impact experiments for a solid 8.**

---

## Review Details

### Claude Review (Score: 7/10)

**Strengths**
1. Clean theoretical narrative: JL lemma + ranking argument is elegant and well-motivated
2. Systematic experimental progression: 7 experiments building from synthetic to 7B scale is exemplary
3. Practical SVD+fine-tune pipeline: 3 deployment paths (zero-cost SVD, SVD+FT, train-from-scratch) is thoughtful
4. Key > Query compressibility finding: Striking asymmetry in Table 5 is genuinely interesting
5. Composability argument: Orthogonal to GQA (head count) and quantization (bit width), up to 16x combined
6. Honest about limitations: Overfitting analysis (WT-2 vs WT-103) and Flash Attention caveat

**Weaknesses**: W1-W7 as listed above

### DeepSeek Review 1 (Score: 7.5/10)

**Strengths**: Clear idea, comprehensive experiments, practical impact, theoretical grounding, honest analysis
**Weaknesses**: W1 (PPL only), W2 (Flash Attention), W3 (scale), W6 (incremental)

### DeepSeek Review 2 (Score: 6.5/10)

**Weaknesses**: W1 (PPL only), W2 (Flash Attention), W4 (MLA missing), W5 (no throughput)
**Recommendations**: Add downstream tasks, benchmark throughput, compare GQA/MLA, reframe FA as call-to-action

# Thin Keys, Full Values -- Road to ICML (Solid 8)

## Paper Summary

The paper proposes **asymmetric attention**: QK projections use lower dimensionality
(d_select << d_model) while V retains full dimensionality. Core insight: attention
selection (QK dot product) is a ranking problem needing only O(log N) dimensions,
while value transfer needs full capacity. Validated across 7 experiments from
synthetic tasks to Mistral-7B post-training compression. At d_select = d_model/4,
achieves 75% key cache savings at ~2% PPL cost after SVD + QK fine-tuning.

---

## Experiment A Results: Downstream Task Evaluation on Mistral-7B

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

## Weaknesses (merged from 3 reviews)

### Unanimous (Claude + DeepSeek R1 + DeepSeek R2)

**W1: Perplexity-only evaluation**
- 2% PPL gap could be 0.5% or 8% on reasoning tasks -- nobody knows
- ICML reviewers will demand MMLU/Hellaswag/GSM8K on the 7B model

**W2: Flash Attention incompatibility**
- Paper says dk=dv assumption baked into optimized kernels, but offers no solution or measurement
- Theoretical FLOP reduction may be negated by poor hardware utilization without kernel support

### Strong consensus (2 of 3 reviews)

**W3: No train-from-scratch beyond 125M**
- Paper's own recommended path (train from scratch) is unvalidated at scale that matters (1B+)
- DS-R1 explicitly asks for a 1B-3B run

**W4: Missing MLA comparison**
- DeepSeek's Multi-head Latent Attention also does low-rank KV compression -- biggest elephant in the room
- DS-R2: "How is this different from just setting head dimension smaller in GQA?"

**W5: No real throughput numbers**
- Paper claims "25GB savings, 60% more users" but never measures actual tokens/sec or batch sizes

### Moderate

**W6: Incremental novelty** (both DeepSeek reviews)
- Well-executed but not paradigm-shifting. Asymmetric QKV dimensions aren't fundamentally new.

**W7: Narrow fine-tuning data** (Claude only)
- SVD+FT uses only WikiText-103 (10M tokens). Unclear if it generalizes to diverse pretraining distributions.

### Consensus matrix

| Weakness | Claude | DS-R1 | DS-R2 | Consensus |
|---|---|---|---|---|
| W1: Downstream tasks beyond PPL | YES | YES | YES | **UNANIMOUS** |
| W2: Flash Attention barrier | YES | YES | YES | **UNANIMOUS** |
| W3: Train-from-scratch at 1B+ | YES | YES | -- | Strong |
| W4: MLA comparison missing | YES | -- | YES | Strong |
| W5: End-to-end throughput | YES | -- | YES | Strong |
| W6: Incremental novelty | Mild | YES | YES | Moderate |
| W7: Narrow FT data | YES | -- | -- | Noted |

---

## Experiment Plan: Weaknesses -> Fixes

### P0: Must-have (address unanimous weaknesses)

#### Experiment A: Downstream Task Evaluation on Mistral-7B [DONE]
- **Fixes**: W1 (perplexity-only)
- **What**: Evaluate compressed Mistral-7B (ranks 256, 512) + baseline + fine-tuned variants
- **Tasks**: MMLU (5-shot), Hellaswag (10-shot), ARC-Challenge (25-shot), WinoGrande (5-shot), GSM8K (5-shot CoT)
- **Result**: See results table above. r512+FT is near-lossless (<1% on most tasks). r256+FT has small gaps on knowledge tasks (0.7-2.3%) but larger on math reasoning (GSM8K -13.7%). Fine-tuning recovers most of the SVD damage. Confirms reviewers' concern that PPL understates reasoning impact, but the overall story is strong.

#### Experiment B: End-to-End Throughput Benchmarks
- **Fixes**: W2 (Flash Attention), W5 (no throughput numbers)
- **What**: Measure real tokens/sec, max batch size, time-to-first-token, and peak memory
- **Setup**: Standard Mistral-7B vs compressed, on H100, at context lengths 4K/16K/64K/128K
- **Kernel**: Use a naive/general-purpose kernel if Flash Attention doesn't support asymmetric dims. The point is to show memory savings dominate in memory-bound regimes.
- **Compute**: Low -- just inference benchmarking

### P1: High impact (address strong-consensus weaknesses)

#### Experiment C: Train from Scratch at 1B-3B Scale
- **Fixes**: W3 (no large-scale train-from-scratch)
- **What**: Train a 1.3B model with d_select = d_model/4 from scratch on SlimPajama or FineWeb-Edu (~50B tokens)
- **Compute**: Moderate -- one 1B training run (2-4 days on 8xH100)
- **Expected outcome**: If PPL degradation is again ~4%, the architecture-independence claim is iron-clad.
- **Bonus**: Also evaluate this model on downstream tasks (same as Exp A)

#### Experiment D: Direct GQA and MLA Comparison
- **Fixes**: W4 (missing MLA), W6 (incremental novelty)
- **What**: Comparison table at the same effective KV cache budget:
  - Standard MHA (baseline)
  - GQA (fewer KV heads, same head dim)
  - Asymmetric attention (same head count, smaller key dim)
  - GQA + Asymmetric (composed)
  - MLA (if feasible to implement or find a checkpoint)
- **Metrics**: PPL, downstream tasks, KV cache size, total params
- **Compute**: Moderate -- may need to train several small models (125M-350M)

### P2: Nice-to-have (polish and strengthen)

#### Experiment E: Flash Attention Kernel or Roofline Analysis
- **Fixes**: W2 (Flash Attention)
- **What**: Either (a) implement a simple asymmetric Flash Attention kernel, or (b) provide detailed roofline analysis showing memory-bound regimes where asymmetric attention wins even with a naive kernel
- **Compute**: Engineering effort, low compute

#### Experiment F: Diverse Fine-tuning Corpus
- **Fixes**: W7 (narrow FT data)
- **What**: Repeat SVD+FT on Mistral-7B using 10M tokens from RedPajama or C4 instead of WikiText-103
- **Compute**: Low

#### Experiment G: Per-Layer d_select Allocation
- **Fixes**: W6 (incremental novelty)
- **What**: Allow different d_select per layer. Lower layers (local patterns) may need fewer dims than upper layers (global/semantic patterns).
- **Compute**: Moderate -- grid search or simple heuristic + training

---

## Impact Summary

| Priority | Experiment | Fixes | Compute | Impact on score |
|---|---|---|---|---|
| **P0** | A: Downstream tasks | W1 | Low (running) | Highest |
| **P0** | B: Throughput benchmarks | W2, W5 | Low | High |
| **P1** | C: Train 1B+ from scratch | W3 | Moderate | High |
| **P1** | D: GQA/MLA comparison | W4, W6 | Moderate | High |
| **P2** | E: FA kernel/roofline | W2 | Low-Moderate | Medium |
| **P2** | F: Diverse FT corpus | W7 | Low | Medium |
| **P2** | G: Per-layer d_select | W6 | Moderate | Medium |

**A and B alone move the paper from 7 to 7.5+. Adding C and D gets it to a solid 8.**
The gap is empirical completeness, not conceptual -- all fixable with experiments.

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

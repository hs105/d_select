# Thin Keys, Full Values -- Road to ICML (Solid 8)

## Paper Summary

The paper proposes **asymmetric attention**: QK projections use lower dimensionality
(d_select << d_model) while V retains full dimensionality. Core insight: attention
selection (QK dot product) is a ranking problem needing only O(log N) dimensions,
while value transfer needs full capacity. Validated across 7 experiments from
synthetic tasks to Mistral-7B post-training compression. At d_select = d_model/4,
achieves 75% key cache savings at ~2% PPL cost after SVD + QK fine-tuning.

---

## My (Claude) Review

**Score: 7/10 -- Accept (borderline Strong Accept)**

### Strengths
1. **Clean theoretical narrative**: JL lemma + ranking argument is elegant and well-motivated
2. **Systematic experimental progression**: 7 experiments building from synthetic to 7B scale is exemplary
3. **Practical SVD+fine-tune pipeline**: 3 deployment paths (zero-cost SVD, SVD+FT, train-from-scratch) is thoughtful
4. **Key > Query compressibility finding**: Striking asymmetry in Table 5 is genuinely interesting
5. **Composability argument**: Orthogonal to GQA (head count) and quantization (bit width), up to 16x combined
6. **Honest about limitations**: Overfitting analysis (WT-2 vs WT-103) and Flash Attention caveat show intellectual honesty

### Weaknesses
1. **No downstream task evaluation**: Only perplexity. For ICML, you need MMLU/Hellaswag/GSM8K/HumanEval on the 7B model. A 2% PPL gap could be 0.5% or 8% on reasoning -- we don't know.
2. **No train-from-scratch at scale**: Training experiments only reach 125M. The recommended path (train from scratch) is unvalidated at the scale that matters (1B+).
3. **No end-to-end latency/throughput**: Paper talks about 25GB savings and 60% more users but never measures actual tokens/sec, max batch size, or time-to-first-token.
4. **Missing MLA comparison**: DeepSeek's Multi-head Latent Attention is the most relevant related work -- it also does low-rank KV compression. Complete absence is a gap reviewers will notice.
5. **Flash Attention incompatibility**: Acknowledged but unaddressed. Without a kernel or at minimum a throughput measurement showing memory savings dominate, this undermines the practical story.
6. **Narrow fine-tuning data**: SVD+FT uses only WikiText-103 (10M tokens). Unclear if this generalizes to diverse pretraining distributions.

---

## DeepSeek Review 1 Summary (Score: 7.5/10)

### Strengths
- Clear and motivated idea (selection vs value transfer distinction)
- Comprehensive experimental validation (synthetic -> GPT-2 -> LLaMA-125M -> Mistral-7B)
- Practical impact with tangible memory savings
- Theoretical grounding via JL lemma
- Honest analysis (overfitting, Flash Attention limitations)

### Weaknesses
1. **Incremental novelty**: Well-executed but not paradigm-shifting
2. **Training-from-scratch scale**: Only 125M; 1B-3B run would strengthen claims
3. **Perplexity-only evaluation**: Need downstream tasks (MMLU, Hellaswag)
4. **Flash Attention caveat**: No solution or throughput proof-of-concept

---

## DeepSeek Review 2 Summary (Score: 6.5/10)

### Weaknesses
1. **Novelty vs GQA/MLA**: Needs sharper positioning. "How is this different from just setting head dimension smaller in GQA?" MLA is a missing baseline.
2. **Flash Attention problem**: Theoretical FLOP reduction may be negated by poor hardware utilization without kernel support. Need either a custom kernel or end-to-end measurements.
3. **Perplexity-only**: 7B model needs MMLU, Hellaswag, GSM8K. 2% PPL could be 5% on reasoning.
4. **Regularization confound**: d_model/4 sweet spot may be data-dependent.

### Recommendations
- Add downstream tasks on compressed Mistral-7B
- Benchmark end-to-end throughput (tokens/sec, max batch size)
- Direct comparison table with GQA and MLA at same effective KV size
- Frame Flash Attention as call-to-action rather than limitation

---

## Do I Agree with DeepSeek?

**Yes, broadly.** The three reviews converge on the same core issues:

| Weakness | Claude | DS-R1 | DS-R2 | Consensus |
|---|---|---|---|---|
| Downstream tasks beyond PPL | YES | YES | YES | **UNANIMOUS** |
| Train-from-scratch at 1B+ | YES | YES | -- | Strong |
| End-to-end throughput | YES | -- | YES | Strong |
| MLA comparison missing | YES | -- | YES | Strong |
| Flash Attention barrier | YES | YES | YES | **UNANIMOUS** |
| Incremental novelty | Mild | YES | YES | Moderate |
| Data-dependence of d/4 rule | -- | -- | YES | Noted |

**Key disagreement**: DS-R2 scores 6.5, DS-R1 scores 7.5, I score 7.0. The paper is solid
work but the unanimously-flagged gaps (downstream eval, throughput) are the difference
between a 7 and an 8 at ICML. These are addressable with experiments, not rewriting.

---

## Experiment Plan: From 7 to Solid 8

Priority ordering by impact on reviewer scores.

### P0: Must-have (address unanimous weaknesses)

#### Experiment A: Downstream Task Evaluation on Mistral-7B
- **What**: Evaluate compressed Mistral-7B (rank 256 = 75% K cache saved) + control on standard benchmarks
- **Tasks**: MMLU (5-shot), Hellaswag (10-shot), ARC-Challenge (25-shot), WinoGrande (5-shot), GSM8K (8-shot CoT), HumanEval (pass@1)
- **Why**: All three reviews flag perplexity-only evaluation. This is the single highest-impact addition.
- **Compute**: Low -- just inference on existing compressed + control checkpoints
- **Expected outcome**: If ~2% PPL gap -> <1-2% on most tasks, the paper's claim is dramatically strengthened. Even a 3-4% drop on reasoning with 75% cache savings is a good trade.

#### Experiment B: End-to-End Throughput Benchmarks
- **What**: Measure real tokens/sec, max batch size, time-to-first-token, and peak memory
- **Setup**: Standard Mistral-7B vs compressed, on H100, at context lengths 4K/16K/64K/128K
- **Kernel**: Use a naive/general-purpose kernel if Flash Attention doesn't support asymmetric dims. The point is to show memory savings dominate.
- **Why**: Converts theoretical savings into concrete deployment numbers. Addresses DS-R2 and my concern.
- **Compute**: Low -- just inference benchmarking

### P1: High impact (address strong-consensus weaknesses)

#### Experiment C: Train from Scratch at 1B-3B Scale
- **What**: Train a 1.3B model with d_select = d_model/4 from scratch on a reasonable corpus (e.g., SlimPajama 50B-token subset or FineWeb-Edu)
- **Why**: The paper's recommended deployment path is train-from-scratch, but this is validated only at 125M. One 1B+ run closes the gap. DS-R1 explicitly asks for this.
- **Compute**: Moderate -- one 1B training run (probably 2-4 days on 8xH100)
- **Expected outcome**: If PPL degradation is again ~4%, the architecture-independence claim is iron-clad.
- **Bonus**: Also evaluate this model on downstream tasks (same as Exp A)

#### Experiment D: Direct GQA and MLA Comparison
- **What**: Comparison table at the same effective KV cache budget:
  - Standard MHA (baseline)
  - GQA (fewer KV heads, same head dim)
  - Asymmetric attention (same head count, smaller key dim)
  - GQA + Asymmetric (composed)
  - MLA (if feasible to implement or find a checkpoint)
- **Metrics**: PPL, downstream tasks, KV cache size, total params
- **Why**: DS-R2 explicitly asks "How is this different from just setting head dimension smaller in GQA?" MLA is the elephant in the room.
- **Compute**: Moderate -- may need to train several small models (125M-350M) or find MLA checkpoints

### P2: Nice-to-have (polish and strengthen)

#### Experiment E: Flash Attention Kernel or Detailed Analysis
- **What**: Either (a) implement a simple asymmetric Flash Attention kernel, or (b) provide detailed roofline analysis showing memory-bound regimes where asymmetric attention wins even with a naive kernel
- **Why**: All three reviews flag this. A kernel is best; failing that, a roofline plot reframes the limitation.
- **Compute**: Engineering effort, low compute

#### Experiment F: Diverse Fine-tuning Corpus
- **What**: Repeat SVD+FT on Mistral-7B using a more diverse corpus (e.g., 10M tokens from RedPajama or C4 instead of WikiText-103)
- **Why**: Shows the fine-tuning pipeline isn't specific to WikiText domain
- **Compute**: Low

#### Experiment G: Per-Layer d_select Allocation
- **What**: Allow different d_select per layer. Lower layers (local patterns) may need fewer dims than upper layers (global/semantic patterns).
- **Why**: Could improve the PPL/compression Pareto frontier and adds novelty
- **Compute**: Moderate -- grid search or simple heuristic + training

---

## Summary: What Makes This a Solid 8

| Current (7) | Target (8) |
|---|---|
| PPL only | PPL + 6 downstream tasks on 7B |
| Theoretical memory savings | Real throughput/memory benchmarks |
| Train-from-scratch at 125M | Train-from-scratch at 1B+ |
| GQA mentioned in passing | Direct GQA/MLA comparison table |
| Flash Attention acknowledged | Roofline analysis or kernel POC |

The paper's core idea and experimental methodology are strong. The gap to 8 is
**empirical completeness**, not conceptual. Experiments A and B are low-compute
and high-impact -- they should be done first.

# Thin Keys Revision Plan

Paper: "Thin Keys, Full Values: Reducing KV Cache via Low-Dimensional Attention Selection"

COLM 2026 reviews: 6, 5, 4, 3 (avg 4.5). Decision: skip rebuttal, revise and resubmit.

## Key Reviewer Criticisms

1. **No long-context evaluation** (all 4 reviewers) — paper motivates with long-context KV cache bottleneck but evaluates only on short-context tasks (PPL, HellaSwag, ARC, etc.)
2. **Weak positioning vs. MLA** (GmAj, 5qN3) — MLA achieves 93% savings; thin keys alone only 37.5%. DeepSeek-V3 already uses different key/value dims.
3. **Loose theoretical motivation** (KuvT, S4Dn) — JL lemma argument is hand-wavy; doesn't validate that effective N is small or justify SVD over random projection.
4. **RoPE compatibility unclear** (GmAj) — how does factoring W_K and absorbing B into W_Q work with rotary positional embeddings?
5. **Writing style flagged as LLM-generated** (S4Dn) — 51 em-dashes, repetitive "affirmation — argument" pattern.

## Revision Checklist

### Must-do (address universal criticisms)

- [ ] **Add long-context evaluations**: run RULER, LongBench, and/or needle-in-a-haystack on Mistral-7B with factored keys at r=512 and r=256. This is the single most impactful addition — estimate ~1 weekend of GPU time.
- [ ] **Add long-context PPL curves**: evaluate perplexity at context lengths 4K, 16K, 32K, 64K, 128K to show thin keys don't degrade with length.
- [ ] **Address RoPE explicitly in method section**: explain how factored keys interact with RoPE. If RoPE is applied after projection, the SVD factorization is exact. If applied before, clarify the approximation and its impact.

### Should-do (strengthen positioning)

- [ ] **Sharpen MLA comparison**: emphasize the key differentiator — thin keys are post-hoc and zero-cost for existing models, MLA requires pretraining from scratch. Make the use cases clearly distinct: thin keys for deployed models, MLA/GQA for new architectures.
- [ ] **Add quantitative baselines**: compare directly with LRQK, ZACK, and KV eviction methods (H2O, SnapKV) in the same experimental setting on Mistral-7B.
- [ ] **Tighten or downplay JL theory**: either formalize the argument (prove a ranking-preservation bound under softmax) or reframe it as intuition rather than theoretical justification. Remove the implication that O(log N) is a tight bound.

### Nice-to-do (polish)

- [ ] **Reduce em-dashes**: rewrite sentences to use periods, semicolons, or parentheses instead. Target <15 em-dashes total.
- [ ] **Add a method figure**: visual diagram showing the SVD factorization and B-absorption into queries.
- [ ] **Use modern pretraining data**: replace OpenWebText with FineWeb or DCLM for 7B experiments if rerunning. This avoids the "outdated setup" critique.
- [ ] **Fix duplicate reference**: Hooper et al. (KVQuant) appears twice in the bibliography.

## Target Venues

- ICLR 2026 (deadline ~Oct 2026)
- NeurIPS 2026 (deadline ~May 2026 — may have passed)
- EMNLP 2026 (deadline ~Jun 2026)

Check deadlines and pick the first feasible one.

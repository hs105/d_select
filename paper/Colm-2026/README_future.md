# Follow-Up Experiments for "Thin Keys, Full Values"

Two planned extensions beyond the CoLM-2026 submission.

---

## 1. Chinchilla-Optimal Scaling Laws (Priority: HIGH)

**Question**: Does the thin-keys advantage hold at truly Chinchilla-optimal
token budgets (~20 tokens/param), or does a cost emerge as the model becomes
capacity-limited?

**Why it matters**: Our 7B/20B result (ratio ~3) shows parity, but Chinchilla-optimal
is ratio ~20. If thin keys remain free at that ratio, the regularization explanation
is fully ruled out. If a small cost emerges, characterizing the scaling law for
d_select is itself a strong contribution.

### Practical plan (~4–5 days wall-clock on 8×H100)

Train thin_keys (d_select=d_model/4) vs full_attn at Chinchilla-optimal ratios
across 3 model sizes. Each pair runs in parallel (4 GPUs each).

| Model | Params | Tokens (20×) | Est. steps | Est. time (4×H100) |
|-------|--------|-------------|-----------|---------------------|
| 125M  | 125M   | 2.5B        | ~5K       | ~2 hours            |
| 350M  | 350M   | 7B          | ~14K      | ~12 hours           |
| 1.3B  | 1.3B   | 26B         | ~50K      | ~3 days             |

Total: ~4 days (1.3B is the bottleneck; 125M and 350M fit in the slack).
Run 2 seeds each for the 125M/350M models (cheap); 1 seed for 1.3B.

For 7B Chinchilla-optimal (140B tokens, ~70 days), extrapolate from the
3-point scaling law fit rather than running directly. Mention existing
7B/20B result as an interpolation point.

### Architecture configs

All models use LLaMA architecture (RMSNorm, SwiGLU, RoPE, no bias):

| Model | d_model | heads | layers | d_ff  | d_select (thin) |
|-------|---------|-------|--------|-------|-----------------|
| 125M  | 768     | 12    | 12     | 2048  | 192              |
| 350M  | 1024    | 16    | 24     | 2816  | 256              |
| 1.3B  | 2048    | 16    | 24     | 5504  | 512              |

### Training details

- Data: OpenWebText (full, 9.9B tokens at `/sg-pretrain/datasets/owt_tokens_full.bin`)
  - 125M: ~0.3 epochs
  - 350M: ~0.7 epochs
  - 1.3B: ~2.6 epochs (repeat data; acceptable per Chinchilla)
- Optimizer: AdamW, cosine LR schedule
- Eval: OWT val PPL, WT103 val PPL, downstream (Hellaswag, ARC-C, MMLU)
- Checkpoints: every 1K steps (125M/350M), every 5K steps (1.3B)

### Expected deliverables

1. **Scaling law plot**: x-axis = model params, y-axis = Δ PPL (thin vs full),
   at Chinchilla-optimal tokens. Fit power law, extrapolate to 7B/70B.
2. **Table**: params, tokens, PPL (full), PPL (thin), Δ%, wall-clock speedup.
3. One-paragraph addition to paper Discussion section.

### Results so far (as of 2026-03-30)

| Scale | Params (full/thin) | Tokens | Δ OWT PPL | Δ WT-103 PPL | Time | Status |
|-------|-------------------|--------|-----------|-------------|------|--------|
| 125M  | 109.5M / 98.9M    | 2.5B   | **+5.4%** | +13.9%      | 1.7h | DONE (seed=42) |
| 350M  | 341.1M / 303.4M   | 7.0B   | **+5.1%** | +14.0%      | 18.8h | DONE (seed=42) |
| 1.3B  | 1279.9M / 1128.9M | 26B    | **+1.1%** (step 38K/99K) | -2.5% | ~69h | RUNNING |
| 7B (Exp C2) | 6.74B / 5.93B | 20B | **+0.1%** | +2.0% | 241h | DONE (2 seeds) |

Key finding: gap is flat ~5% from 125M→350M, then drops sharply at 1.3B (~1%).
The transition happens in the 350M–1.3B range.

### Remaining work

1. **Finish 1.3B** — ETA ~April 1 mid-day
2. **Run 125M + 350M with seed=137** for error bars (mean ± std over 2 seeds) — ~14h total
3. **Downstream evals** on all finished checkpoints (Hellaswag, ARC-C, MMLU) — ~2-4h
4. **Write up scaling law section** — fit power law, extrapolate to 7B/70B
5. Learned Factored Keys only if time permits (lower priority now)

### Launch order

1. ~~Start 125M pair first (smoke test, ~2 hours)~~ DONE
2. ~~Start 350M pair (verify, ~12 hours)~~ DONE
3. ~~Start 1.3B pair (bottleneck, ~3 days)~~ RUNNING
4. Run 125M seed=137 + 350M seed=137 after 1.3B finishes (~14h)
5. Downstream evals on all checkpoints (~2-4h)

---

## 2. Learned Factored Keys (Beyond SVD) (Priority: LOW — only if time permits)

**Question**: Can we beat SVD+QK-finetune by learning the low-rank factorization
W_K = A·B end-to-end, rather than fixing A from SVD and only fine-tuning Q?

**Why it matters**: SVD gives the optimal rank-r approximation in Frobenius norm,
but not in the metric that matters (downstream loss). At aggressive compression
(r=128 on 7B, currently 3.2% cost), a learned factorization could close the gap.
Lower priority now — the scaling law curve is the bigger story.

### Plan (~1 day wall-clock)

Apply to Mistral-7B, reusing the Experiment F/F3 fine-tuning pipeline.

**Variants to compare** (all initialized from SVD):

| Variant | What's trained | Description |
|---------|---------------|-------------|
| Baseline (current) | Q only | SVD-fix K factorization, fine-tune W_Q' |
| Learned-AB | A, B, Q | Train both factors of W_K plus W_Q |
| LoRA-residual | Q + LoRA(K) | SVD-fix K, add LoRA adapter on residual |

**Configs**: r ∈ {128, 256, 512} × 3 variants = 9 runs.
Each run: 3 epochs on WikiText-103 (~10M tokens), ~1–2 hours on 1 GPU.
All 9 runs parallelize across 8 GPUs → **~3 hours total**.

Then evaluate best configs on GSM8K CoT (Exp F3 protocol) for another ~2 hours.

### Training details

- Fine-tune data: WikiText-103 (same as Exp A/F)
- LR: 5e-5, AdamW, cosine schedule, 3 epochs
- For Learned-AB: may need lower LR on B to avoid divergence (try 1e-5)
- For LoRA-residual: rank-16 or rank-32 LoRA on the K residual

### Expected deliverables

1. **Table**: rank × variant, PPL before/after, vs control
2. **Bar chart**: compression gap (%) at r=128 across variants
3. If Learned-AB wins, add as a 4th deployment path in Discussion:
   "SVD + end-to-end QK fine-tuning" between paths (2) and (3).

---

## Execution Order

1. ~~**Chinchilla Scaling Laws**~~ — IN PROGRESS, 125M+350M done, 1.3B running
2. **Seed=137 reruns** for error bars (~14h) — after 1.3B finishes
3. **Downstream evals** on all checkpoints (~2-4h)
4. **Write up** scaling law results
5. **Learned Factored Keys** (~1 day) — only if time permits

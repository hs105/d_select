#!/bin/bash
# Run K-only SVD compression experiments on GPT-2
# Also runs Q-only and Both for comparison
#
# Usage:
#   chmod +x run_compress.sh
#   ./run_compress.sh [GPU_ID]
#
# Default GPU: 0

GPU=${1:-0}
cd "$(dirname "$0")"
mkdir -p logs

echo "=== K-only, Q-only, and Both SVD compression on GPT-2 ==="
echo "GPU: $GPU"
echo ""

CUDA_VISIBLE_DEVICES=$GPU python -u -c "
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from compress_qk import load_wikitext2, evaluate_perplexity, svd_compress_weight

device = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
eval_ids = load_wikitext2(tokenizer, 1024)

# Baseline
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
_, base_ppl = evaluate_perplexity(model, eval_ids, device)
print(f'Baseline: {base_ppl:.2f}', flush=True)
del model
torch.cuda.empty_cache()

# K-only sweep
print('\n=== K-ONLY ===', flush=True)
for rank in [128, 192, 256, 384, 512]:
    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
        for i in range(12):
            w = model.transformer.h[i].attn.c_attn.weight.data
            W_K = w[:, 768:1536]
            W_K_c, err, _, _ = svd_compress_weight(W_K, rank)
            w[:, 768:1536] = W_K_c
        _, ppl = evaluate_perplexity(model, eval_ids, device)
        delta = (ppl - base_ppl) / base_ppl * 100
        print(f'K-only rank={rank}: PPL={ppl:.2f} ({delta:+.1f}%), err={err:.4f}', flush=True)
    except Exception as e:
        print(f'K-only rank={rank}: FAILED -- {e}', flush=True)
    del model
    torch.cuda.empty_cache()

# Q-only sweep
print('\n=== Q-ONLY ===', flush=True)
for rank in [128, 192, 256, 384, 512]:
    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
        for i in range(12):
            w = model.transformer.h[i].attn.c_attn.weight.data
            W_Q = w[:, :768]
            W_Q_c, err, _, _ = svd_compress_weight(W_Q, rank)
            w[:, :768] = W_Q_c
        _, ppl = evaluate_perplexity(model, eval_ids, device)
        delta = (ppl - base_ppl) / base_ppl * 100
        print(f'Q-only rank={rank}: PPL={ppl:.2f} ({delta:+.1f}%), err={err:.4f}', flush=True)
    except Exception as e:
        print(f'Q-only rank={rank}: FAILED -- {e}', flush=True)
    del model
    torch.cuda.empty_cache()

# Both Q+K sweep
print('\n=== BOTH Q+K ===', flush=True)
for rank in [128, 192, 256, 384, 512]:
    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
        for i in range(12):
            w = model.transformer.h[i].attn.c_attn.weight.data
            W_Q = w[:, :768]
            W_K = w[:, 768:1536]
            W_Q_c, q_err, _, _ = svd_compress_weight(W_Q, rank)
            W_K_c, k_err, _, _ = svd_compress_weight(W_K, rank)
            w[:, :768] = W_Q_c
            w[:, 768:1536] = W_K_c
        _, ppl = evaluate_perplexity(model, eval_ids, device)
        delta = (ppl - base_ppl) / base_ppl * 100
        print(f'Both rank={rank}: PPL={ppl:.2f} ({delta:+.1f}%), Q_err={q_err:.4f}, K_err={k_err:.4f}', flush=True)
    except Exception as e:
        print(f'Both rank={rank}: FAILED -- {e}', flush=True)
    del model
    torch.cuda.empty_cache()

print('\n=== DONE ===', flush=True)
" 2>&1 | tee logs/svd_all_modes.log
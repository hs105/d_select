

cd /sg-pretrain/focus

CUDA_VISIBLE_DEVICES=0 python -u -c "
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from compress_qk import load_wikitext2, evaluate_perplexity, svd_compress_weight

device = torch.device('cuda')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
eval_ids = load_wikitext2(tokenizer, 1024)

print('=== Q-ONLY SVD COMPRESSION ===', flush=True)
for rank in [128, 192, 256, 384, 512]:
    try:
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device).eval()
        for i in range(12):
            w = model.transformer.h[i].attn.c_attn.weight.data
            W_Q = w[:, :768]
            W_Q_c, err, _, _ = svd_compress_weight(W_Q, rank)
            w[:, :768] = W_Q_c
        _, ppl = evaluate_perplexity(model, eval_ids, device)
        delta = (ppl - 24.91) / 24.91 * 100
        print(f'Q-only rank={rank}: PPL={ppl:.2f} ({delta:+.1f}%), err={err:.4f}', flush=True)
    except Exception as e:
        print(f'Q-only rank={rank}: FAILED -- {e}', flush=True)
    del model
    torch.cuda.empty_cache()
print('=== DONE ===', flush=True)
"
"""
SVD Compress K + Fine-tune for Mistral-7B
==========================================
Mistral-7B architecture:
  d_model = 4096
  n_heads = 32 (query), n_kv_heads = 8 (GQA)
  d_head = 128
  W_K: [4096, 1024] (8 heads × 128 dims)
  W_Q: [4096, 4096] (32 heads × 128 dims)
  32 layers

Strategy:
  1. Load Mistral-7B
  2. Evaluate baseline PPL on WikiText-103 test
  3. SVD compress W_K per layer to target rank
  4. Evaluate compressed PPL
  5. Fine-tune QK projections only
  6. Evaluate recovered PPL
"""

import argparse
import gc
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Data
# ============================================================
def load_wikitext103_split(tokenizer, split, seq_len=2048, max_tokens=None):
    """Load WikiText-103 split."""
    split_map = {
        'train': 'wiki.train.tokens',
        'validation': 'wiki.valid.tokens',
        'test': 'wiki.test.tokens',
    }
    fpath = f'/root/data/wikitext-103/{split_map[split]}'
    print(f"  Loading {fpath}...", flush=True)
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()

    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'][0]
    total_tokens = len(input_ids)

    if max_tokens and len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]
        print(f"  Truncated to {max_tokens:,} tokens")

    n_chunks = len(input_ids) // seq_len
    input_ids = input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)
    print(f"  {split}: {total_tokens:,} total tokens, "
          f"using {n_chunks * seq_len:,} ({n_chunks} chunks of {seq_len})", flush=True)
    return input_ids


# ============================================================
# Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model, input_ids, device, batch_size=1):
    model.eval()
    total_loss = 0
    n_chunks = input_ids.shape[0]

    for i in range(0, n_chunks, batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        outputs = model(batch, labels=batch)
        total_loss += outputs.loss.item() * batch.shape[0]

        if (i // batch_size) % 50 == 0:
            print(f"    eval {i}/{n_chunks}...", end='\r', flush=True)

    avg_loss = total_loss / n_chunks
    ppl = math.exp(min(avg_loss, 20))
    print(f"    eval done.           ", flush=True)
    return avg_loss, ppl


# ============================================================
# SVD compression of W_K
# ============================================================
def compress_k_layers(model, rank, verbose=True):
    """
    Compress W_K in all layers via SVD.
    Mistral: model.layers[i].self_attn.k_proj.weight is [1024, 4096]
    (PyTorch stores as [out_features, in_features])
    """
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    k_dim = n_kv_heads * d_head  # 8 * 128 = 1024

    if verbose:
        print(f"  K projection: [{k_dim}, {d_model}] (out, in)")
        print(f"  Target rank: {rank} (of {k_dim})")

    errors = []
    for i in range(n_layers):
        W_K = model.model.layers[i].self_attn.k_proj.weight.data.float()
        # W_K shape: [1024, 4096]

        U, S, Vh = torch.linalg.svd(W_K, full_matrices=False)
        # U: [1024, 1024], S: [1024], Vh: [1024, 4096]

        W_K_compressed = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
        err = torch.norm(W_K - W_K_compressed).item() / torch.norm(W_K).item()
        errors.append(err)

        model.model.layers[i].self_attn.k_proj.weight.data = W_K_compressed.to(
            model.model.layers[i].self_attn.k_proj.weight.dtype
        )

        if verbose and (i == 0 or i == n_layers - 1 or (i + 1) % 8 == 0):
            print(f"    Layer {i:2d}: K error = {err:.4f}")

    avg_err = sum(errors) / len(errors)
    if verbose:
        print(f"    Average K error: {avg_err:.4f}")
    return errors


# ============================================================
# Fine-tuning
# ============================================================
def finetune_epoch(model, train_ids, optimizer, scheduler, device,
                   batch_size=1, grad_accum=8, grad_clip=1.0):
    model.train()
    total_loss = 0
    n_chunks = train_ids.shape[0]
    start = time.time()

    perm = torch.randperm(n_chunks)
    optimizer.zero_grad()

    for step in range(0, n_chunks, batch_size):
        idx = perm[step:step+batch_size]
        batch = train_ids[idx].to(device)

        outputs = model(batch, labels=batch)
        loss = outputs.loss / grad_accum
        loss.backward()

        total_loss += outputs.loss.item() * batch.shape[0]

        if (step // batch_size + 1) % grad_accum == 0 or step + batch_size >= n_chunks:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        if (step // batch_size) % 100 == 0:
            print(f"    train step {step}/{n_chunks}...", end='\r', flush=True)

    elapsed = time.time() - start
    avg_loss = total_loss / n_chunks
    ppl = math.exp(min(avg_loss, 20))
    tok_s = n_chunks * train_ids.shape[1] / max(elapsed, 1e-6)
    return avg_loss, ppl, tok_s


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/sg-pretrain/models/mistral-7b')
    parser.add_argument('--rank', type=int, default=256,
                        help='SVD rank for K compression (K dim is 1024)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--max_train_tokens', type=int, default=10_000_000)
    parser.add_argument('--max_eval_tokens', type=int, default=None,
                        help='Cap eval tokens for faster iteration')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str,
                        default='/sg-pretrain/focus/checkpoints_7b')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print(f"SVD Compress K (rank={args.rank}) + Fine-tune Mistral-7B")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load data
    print("\nLoading WikiText-103...", flush=True)
    train_ids = load_wikitext103_split(
        tokenizer, 'train', args.seq_len, args.max_train_tokens)
    val_ids = load_wikitext103_split(
        tokenizer, 'validation', args.seq_len, args.max_eval_tokens)
    test_ids = load_wikitext103_split(
        tokenizer, 'test', args.seq_len, args.max_eval_tokens)

    # Load model
    print("\nLoading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    d_model = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_kv_heads = model.config.num_key_value_heads
    d_head = d_model // n_heads
    k_dim = n_kv_heads * d_head
    n_layers = model.config.num_hidden_layers

    print(f"  d_model={d_model}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
    print(f"  d_head={d_head}, K dim={k_dim}, layers={n_layers}")
    print(f"  Target rank: {args.rank} ({args.rank}/{k_dim} = "
          f"{args.rank/k_dim:.1%} of K dim)")
    print(f"  Model dtype: {model.dtype}")

    # Baseline
    print("\nBaseline (uncompressed)...", flush=True)
    _, base_ppl = evaluate(model, test_ids, device, args.batch_size)
    print(f"  Test PPL: {base_ppl:.2f}", flush=True)

    # Compress K
    print(f"\nCompressing W_K to rank {args.rank}...", flush=True)
    compress_k_layers(model, args.rank)
    torch.cuda.empty_cache()

    _, compressed_ppl = evaluate(model, test_ids, device, args.batch_size)
    delta = (compressed_ppl - base_ppl) / base_ppl * 100
    print(f"\nAfter compression (before fine-tuning):", flush=True)
    print(f"  Test PPL: {compressed_ppl:.2f} ({delta:+.1f}%)", flush=True)

    # Set up optimizer — QK projections only
    params = []
    for layer in model.model.layers:
        params.append({'params': list(layer.self_attn.q_proj.parameters())})
        params.append({'params': list(layer.self_attn.k_proj.parameters())})

    n_ft = sum(p.numel() for pg in params for p in pg['params'])
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\nFine-tuning: QK projections only "
          f"({n_ft:,} of {n_total:,} params, {n_ft/n_total:.1%})", flush=True)

    # Freeze everything except QK
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.model.layers:
        for p in layer.self_attn.q_proj.parameters():
            p.requires_grad = True
        for p in layer.self_attn.k_proj.parameters():
            p.requires_grad = True

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    steps_per_epoch = train_ids.shape[0] // (args.batch_size * args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Fine-tune
    os.makedirs(args.save_dir, exist_ok=True)
    gap = compressed_ppl - base_ppl

    print(f"\nFine-tuning for {args.epochs} epochs "
          f"(lr={args.lr}, grad_accum={args.grad_accum})...", flush=True)
    print("-" * 80)
    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Train PPL':>10} {'tok/s':>8} | "
          f"{'Val PPL':>10} {'ΔPPL':>8} | {'vs ctrl':>8}")
    print("-" * 80)

    best_val_ppl = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl, tok_s = finetune_epoch(
            model, train_ids, optimizer, scheduler, device,
            args.batch_size, args.grad_accum, args.grad_clip
        )
        _, val_ppl = evaluate(model, val_ids, device, args.batch_size)
        delta_val = (val_ppl - base_ppl) / base_ppl * 100

        print(f"{epoch:5d} | {train_loss:10.4f} {train_ppl:10.2f} {tok_s:>7.0f} | "
              f"{val_ppl:10.2f} {delta_val:>+7.1f}% |", flush=True)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl

    print("-" * 80)

    # Final test
    _, test_ppl = evaluate(model, test_ids, device, args.batch_size)
    test_delta = (test_ppl - base_ppl) / base_ppl * 100

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline (uncompressed):      {base_ppl:.2f}")
    print(f"After SVD rank={args.rank}:        {compressed_ppl:.2f} "
          f"({(compressed_ppl-base_ppl)/base_ppl*100:+.1f}%)")
    print(f"After fine-tuning ({args.epochs}ep):    {test_ppl:.2f} "
          f"({test_delta:+.1f}%)")
    if gap > 0:
        recovered = (compressed_ppl - test_ppl) / gap * 100
        print(f"Gap recovered:                {recovered:.1f}%")
    print(f"\nK dim: {k_dim}, rank: {args.rank}, "
          f"K cache reduction: {1 - args.rank/k_dim:.0%}")

    # Save
    results = {
        'model': 'mistral-7b',
        'rank': args.rank,
        'k_dim': k_dim,
        'base_ppl': float(base_ppl),
        'compressed_ppl': float(compressed_ppl),
        'finetuned_ppl': float(test_ppl),
        'delta_pct': float(test_delta),
        'epochs': args.epochs,
        'lr': args.lr,
        'n_ft_params': n_ft,
        'n_total_params': n_total,
    }
    rpath = os.path.join(args.save_dir, f'mistral7b_svd_r{args.rank}.json')
    with open(rpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {rpath}")


if __name__ == '__main__':
    main()
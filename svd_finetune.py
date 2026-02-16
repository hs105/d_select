"""
SVD Compress K + Fine-tune GPT-2
=================================
1. Load pretrained GPT-2
2. SVD compress W_K to target rank
3. Fine-tune on WikiText-2 for a few epochs
4. Measure PPL recovery

Tests whether continued training can close the gap between
post-training SVD (+26% at rank 192) and training from scratch (~4%).
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from compress_qk import svd_compress_weight


# ============================================================
# Data — uses same approach as compress_qk.py
# ============================================================
def load_wikitext2_split(tokenizer, split, seq_len=1024):
    """Load a WikiText-2 split using HuggingFace datasets."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=split)
    text = "\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'][0]
    n_chunks = len(input_ids) // seq_len
    input_ids = input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)
    print(f"  {split}: {len(encodings['input_ids'][0]):,} tokens, {n_chunks} chunks")
    return input_ids


def load_wikitext103_split(tokenizer, split, seq_len=1024, max_train_tokens=None):
    """Load a WikiText-103 split from local files."""
    split_map = {
        'train': 'wiki.train.tokens',
        'validation': 'wiki.valid.tokens',
        'test': 'wiki.test.tokens',
    }
    fpath = f'/root/data/wikitext-103/{split_map[split]}'
    print(f"  Loading {fpath}...")
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'][0]
    if max_train_tokens and split == 'train' and len(input_ids) > max_train_tokens:
        input_ids = input_ids[:max_train_tokens]
        print(f"  Truncated train to {max_train_tokens:,} tokens")
    n_chunks = len(input_ids) // seq_len
    input_ids = input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)
    print(f"  {split}: {len(input_ids) * seq_len:,} tokens, {n_chunks} chunks")
    return input_ids


# ============================================================
# Evaluation — same as compress_qk.py
# ============================================================
@torch.no_grad()
def evaluate(model, input_ids, device, batch_size=8):
    model.eval()
    total_loss = 0
    n_chunks = input_ids.shape[0]

    for i in range(0, n_chunks, batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        outputs = model(batch, labels=batch)
        total_loss += outputs.loss.item() * batch.shape[0]

    avg_loss = total_loss / n_chunks
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


# ============================================================
# SVD compression of W_K only
# ============================================================
def compress_k_only(model, rank, verbose=True):
    n_layers = model.config.n_layer
    d_model = model.config.n_embd
    errors = []

    for i in range(n_layers):
        w = model.transformer.h[i].attn.c_attn.weight.data
        W_K = w[:, d_model:2*d_model]
        W_K_compressed, err, _, _ = svd_compress_weight(W_K, rank)
        w[:, d_model:2*d_model] = W_K_compressed
        errors.append(err)

        if verbose and (i == 0 or i == n_layers - 1):
            print(f"  Layer {i}: K reconstruction error = {err:.4f}")

    avg_err = sum(errors) / len(errors)
    if verbose:
        print(f"  Average K error: {avg_err:.4f}")
    return errors


# ============================================================
# Fine-tuning
# ============================================================
def finetune_epoch(model, train_ids, optimizer, scheduler, device,
                   batch_size=8, grad_clip=1.0):
    model.train()
    total_loss = 0
    n_chunks = train_ids.shape[0]
    start = time.time()

    # Shuffle
    perm = torch.randperm(n_chunks)

    for i in range(0, n_chunks, batch_size):
        idx = perm[i:i+batch_size]
        batch = train_ids[idx].to(device)

        outputs = model(batch, labels=batch)
        loss = outputs.loss

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item() * batch.shape[0]

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
    parser.add_argument('--rank', type=int, default=192)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--finetune_mode', type=str, default='all',
                        choices=['all', 'qk_only', 'attn_only'])
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2', 'wikitext103'])
    parser.add_argument('--max_train_tokens', type=int, default=None,
                        help='Cap training tokens (useful for wikitext103)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_finetune')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print(f"SVD Compress K (rank={args.rank}) + Fine-tune GPT-2")
    print("=" * 70)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load data
    print(f"\nLoading {args.dataset}...")
    if args.dataset == 'wikitext103':
        train_ids = load_wikitext103_split(tokenizer, 'train', args.seq_len,
                                            args.max_train_tokens)
        val_ids = load_wikitext103_split(tokenizer, 'validation', args.seq_len)
        test_ids = load_wikitext103_split(tokenizer, 'test', args.seq_len)
    else:
        train_ids = load_wikitext2_split(tokenizer, 'train', args.seq_len)
        val_ids = load_wikitext2_split(tokenizer, 'validation', args.seq_len)
        test_ids = load_wikitext2_split(tokenizer, 'test', args.seq_len)

    # Load model
    print("\nLoading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    d_model = model.config.n_embd
    n_heads = model.config.n_head
    print(f"  d_model={d_model}, n_heads={n_heads}, "
          f"rank={args.rank} ({args.rank//n_heads}/head)")

    # Baseline
    print("\nBaseline (uncompressed)...", flush=True)
    _, base_ppl = evaluate(model, test_ids, device, args.batch_size)
    print(f"  Test PPL: {base_ppl:.2f}", flush=True)

    # Compress K only
    print(f"\nCompressing W_K to rank {args.rank}...", flush=True)
    compress_k_only(model, args.rank)

    _, compressed_ppl = evaluate(model, test_ids, device, args.batch_size)
    delta = (compressed_ppl - base_ppl) / base_ppl * 100
    print(f"\nAfter compression (before fine-tuning):", flush=True)
    print(f"  Test PPL: {compressed_ppl:.2f} ({delta:+.1f}%)", flush=True)

    # Optimizer setup
    if args.finetune_mode == 'qk_only':
        params = []
        for layer in model.transformer.h:
            params.append({'params': list(layer.attn.c_attn.parameters())})
        n_ft = sum(p.numel() for pg in params for p in pg['params'])
        print(f"\nFine-tuning: QK projections only ({n_ft:,} params)")
    elif args.finetune_mode == 'attn_only':
        params = []
        for layer in model.transformer.h:
            params.append({'params': list(layer.attn.parameters())})
        n_ft = sum(p.numel() for pg in params for p in pg['params'])
        print(f"\nFine-tuning: attention only ({n_ft:,} params)")
    else:
        params = model.parameters()
        n_ft = sum(p.numel() for p in model.parameters())
        print(f"\nFine-tuning: all parameters ({n_ft:,} params)")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    total_steps = (train_ids.shape[0] // args.batch_size) * args.epochs
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
          f"(lr={args.lr}, mode={args.finetune_mode})...", flush=True)
    print("-" * 80)
    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Train PPL':>10} {'tok/s':>8} | "
          f"{'Val PPL':>10} {'ΔPPL':>8} | {'Recovered':>10}")
    print("-" * 80)

    best_val_ppl = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl, tok_s = finetune_epoch(
            model, train_ids, optimizer, scheduler, device,
            args.batch_size, args.grad_clip
        )
        _, val_ppl = evaluate(model, val_ids, device, args.batch_size)
        delta_val = (val_ppl - base_ppl) / base_ppl * 100
        recovered = max(0, (compressed_ppl - val_ppl) / gap * 100) if gap > 0 else 0

        print(f"{epoch:5d} | {train_loss:10.4f} {train_ppl:10.2f} {tok_s:>7.0f} | "
              f"{val_ppl:10.2f} {delta_val:>+7.1f}% | {recovered:>9.1f}%", flush=True)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl

    print("-" * 80)

    # Final test
    _, test_ppl = evaluate(model, test_ids, device, args.batch_size)
    test_delta = (test_ppl - base_ppl) / base_ppl * 100
    total_recovered = max(0, (compressed_ppl - test_ppl) / gap * 100) if gap > 0 else 0

    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Baseline (uncompressed):      {base_ppl:.2f}")
    print(f"After SVD rank={args.rank}:        {compressed_ppl:.2f} ({(compressed_ppl-base_ppl)/base_ppl*100:+.1f}%)")
    print(f"After fine-tuning ({args.epochs}ep):    {test_ppl:.2f} ({test_delta:+.1f}%)")
    print(f"Gap recovered:                {total_recovered:.1f}%")
    print(f"")
    print(f"Target: <2% degradation = PPL < {base_ppl * 1.02:.2f}")
    if test_ppl < base_ppl * 1.02:
        print(f">>> SUCCESS: {test_ppl:.2f} < {base_ppl * 1.02:.2f}")
    elif test_ppl < base_ppl * 1.05:
        print(f">>> CLOSE: {test_ppl:.2f} (within 5%)")
    else:
        print(f">>> NEEDS MORE WORK: {test_ppl:.2f}")

    # Save
    results = {
        'rank': args.rank,
        'base_ppl': base_ppl,
        'compressed_ppl': compressed_ppl,
        'finetuned_ppl': test_ppl,
        'delta_pct': test_delta,
        'gap_recovered_pct': total_recovered,
        'epochs': args.epochs,
        'lr': args.lr,
        'finetune_mode': args.finetune_mode,
    }
    rpath = os.path.join(args.save_dir, f'svd_finetune_r{args.rank}_{args.dataset}.json')
    with open(rpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {rpath}")


if __name__ == '__main__':
    main()
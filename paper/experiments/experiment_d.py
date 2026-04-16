"""
Experiment D: GQA vs MLA vs Thin Keys — Train-from-Scratch Comparison
======================================================================
Trains 125M LLaMA models on WikiText-103 with different attention mechanisms.
Produces a comparison of KV cache budget vs perplexity.

Addresses reviewer DS-R2: "How is this different from just setting the
head dimension smaller in GQA?"

Methods:
  mha        — Standard Multi-Head Attention (baseline)
  gqa        — Grouped-Query Attention (reduce KV head count)
  thin_keys  — This paper's method (reduce QK dim, keep V full)
  mla        — Multi-head Latent Attention (shared low-rank KV latent)

Usage:
  CUDA_VISIBLE_DEVICES=3 python experiment_d.py --method mha
  CUDA_VISIBLE_DEVICES=4 python experiment_d.py --method gqa --n_kv_heads 4
  CUDA_VISIBLE_DEVICES=5 python experiment_d.py --method thin_keys --d_select 192
  CUDA_VISIBLE_DEVICES=6 python experiment_d.py --method mla --d_compressed 384
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from llama_model import (RMSNorm, RotaryEmbedding, apply_rotary_emb, SwiGLU,
                          AsymmetricLlamaAttention)
from train_llama import (load_wikitext, SimpleTokenizer, TextDataset,
                          train_epoch, evaluate)


# ============================================================
# GQA Attention
# ============================================================
class GQAAttention(nn.Module):
    """Grouped-Query Attention (Ainslie et al., 2023).

    Uses fewer KV heads than query heads. Each KV head is shared across
    a group of query heads. KV cache = n_kv_heads * d_head * 2.
    """
    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len=4096, bias=False):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Linear(d_model, n_heads * self.d_head, bias=bias)
        self.W_K = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.W_V = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RotaryEmbedding(self.d_head, max_seq_len=max_seq_len)

    def forward(self, H, attn_mask=None):
        B, N, D = H.shape

        Q = self.W_Q(H).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(H).view(B, N, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_V(H).view(B, N, self.n_kv_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(N)
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        # Expand KV heads to match query head groups
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is None:
            attn_mask = torch.tril(
                torch.ones(N, N, device=H.device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(~attn_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_O(out)


# ============================================================
# MLA Attention
# ============================================================
class MLAAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2, 2024).

    Projects input to a shared low-rank latent, then up-projects to K and V.
    At inference, only the d_compressed-dim latent is cached per token.
    KV cache = d_compressed elements per token per layer.
    """
    def __init__(self, d_model, n_heads, d_compressed, max_seq_len=4096, bias=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_compressed = d_compressed

        self.W_D = nn.Linear(d_model, d_compressed, bias=bias)
        self.W_UK = nn.Linear(d_compressed, n_heads * self.d_head, bias=bias)
        self.W_UV = nn.Linear(d_compressed, n_heads * self.d_head, bias=bias)
        self.W_Q = nn.Linear(d_model, n_heads * self.d_head, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        self.rope = RotaryEmbedding(self.d_head, max_seq_len=max_seq_len)

    def forward(self, H, attn_mask=None):
        B, N, D = H.shape

        c = self.W_D(H)
        K = self.W_UK(c).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_UV(c).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        Q = self.W_Q(H).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        cos, sin = self.rope(N)
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is None:
            attn_mask = torch.tril(
                torch.ones(N, N, device=H.device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(~attn_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_O(out)


# ============================================================
# Generic LLaMA model with pluggable attention
# ============================================================
class GenericBlock(nn.Module):
    def __init__(self, d_model, d_ff, attn):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = attn
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, H):
        H = H + self.attn(self.attn_norm(H))
        H = H + self.ffn(self.ffn_norm(H))
        return H


class ComparisonLlama(nn.Module):
    def __init__(self, vocab_size, d_model, blocks, max_seq_len=512,
                 tie_weights=True):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(blocks)
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        B, N = input_ids.shape
        H = self.tok_emb(input_ids)
        for layer in self.layers:
            H = layer(H)
        H = self.norm(H)
        logits = self.lm_head(H)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1), ignore_index=-1)
        return logits, loss


# ============================================================
# Model configs and builders
# ============================================================
MODEL_CONFIGS = {
    '25M':  dict(d_model=512,  n_heads=8,  n_layers=8,  d_ff=1376),
    '50M':  dict(d_model=512,  n_heads=8,  n_layers=16, d_ff=1376),
    '125M': dict(d_model=768,  n_heads=12, n_layers=12, d_ff=2048),
    '350M': dict(d_model=1024, n_heads=16, n_layers=24, d_ff=2816),
}


def build_model(method, vocab_size, d_model, n_heads, n_layers, d_ff,
                max_seq_len, **kwargs):
    blocks = []
    for _ in range(n_layers):
        if method == 'mha':
            attn = AsymmetricLlamaAttention(
                d_model, n_heads, d_model, max_seq_len)
        elif method == 'gqa':
            attn = GQAAttention(
                d_model, n_heads, kwargs['n_kv_heads'], max_seq_len)
        elif method == 'thin_keys':
            attn = AsymmetricLlamaAttention(
                d_model, n_heads, kwargs['d_select'], max_seq_len)
        elif method == 'mla':
            attn = MLAAttention(
                d_model, n_heads, kwargs['d_compressed'], max_seq_len)
        else:
            raise ValueError(f"Unknown method: {method}")
        blocks.append(GenericBlock(d_model, d_ff, attn))
    return ComparisonLlama(vocab_size, d_model, blocks, max_seq_len)


def kv_cache_budget(method, d_model, n_heads, **kwargs):
    """KV cache elements per token per layer."""
    d_head = d_model // n_heads
    if method == 'mha':
        return n_heads * d_head * 2
    elif method == 'gqa':
        return kwargs['n_kv_heads'] * d_head * 2
    elif method == 'thin_keys':
        return kwargs['d_select'] + d_model
    elif method == 'mla':
        return kwargs['d_compressed']


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Experiment D: GQA vs MLA vs Thin Keys comparison')
    parser.add_argument('--method', required=True,
                        choices=['mha', 'gqa', 'thin_keys', 'mla'])
    parser.add_argument('--n_kv_heads', type=int, default=4)
    parser.add_argument('--d_select', type=int, default=192)
    parser.add_argument('--d_compressed', type=int, default=384)
    parser.add_argument('--size', type=str, default='125M',
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--data_path', type=str, default='/root/data')
    parser.add_argument('--min_freq', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.size]
    d_model, n_heads = cfg['d_model'], cfg['n_heads']
    n_layers, d_ff = cfg['n_layers'], cfg['d_ff']

    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(__file__), 'logs')

    # Method-specific kwargs and run name
    mkw = {}
    if args.method == 'gqa':
        mkw['n_kv_heads'] = args.n_kv_heads
        tag = f"gqa{args.n_kv_heads}"
    elif args.method == 'thin_keys':
        mkw['d_select'] = args.d_select
        tag = f"thin{args.d_select}"
    elif args.method == 'mla':
        mkw['d_compressed'] = args.d_compressed
        tag = f"mla{args.d_compressed}"
    else:
        tag = "mha"

    run_name = f"expD_{args.size}_{tag}"

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    kv = kv_cache_budget(args.method, d_model, n_heads, **mkw)
    kv_base = n_heads * (d_model // n_heads) * 2
    kv_pct = 1 - kv / kv_base

    print("=" * 70)
    print(f"Experiment D: {run_name}")
    print("=" * 70)
    print(f"Method:    {args.method} {mkw}")
    print(f"Model:     {args.size} (d={d_model}, heads={n_heads}, layers={n_layers})")
    print(f"KV cache:  {kv} elem/tok/layer (baseline={kv_base}, saving={kv_pct:.1%})")
    print()

    # ---- Data ----
    print("Loading data...")
    wt = load_wikitext(args.data_path)
    if wt is None:
        print("ERROR: No WikiText data found at", args.data_path)
        return

    train_texts = [l.strip() for l in wt['train'].split('\n')
                   if l.strip() and not l.strip().startswith('=')]
    eval_texts = [l.strip() for l in wt['valid'].split('\n')
                  if l.strip() and not l.strip().startswith('=')]
    test_texts = [l.strip() for l in wt['test'].split('\n')
                  if l.strip() and not l.strip().startswith('=')]

    tokenizer = SimpleTokenizer(min_freq=args.min_freq)
    tokenizer.build_vocab(train_texts)

    train_tokens = tokenizer.encode_texts(train_texts)
    eval_tokens = tokenizer.encode_texts(eval_texts)
    test_tokens = tokenizer.encode_texts(test_texts)

    train_dataset = TextDataset(train_tokens, args.max_seq_len)
    eval_dataset = TextDataset(eval_tokens, args.max_seq_len)
    test_dataset = TextDataset(test_tokens, args.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    print(f"Train: {len(train_dataset)} seqs, Val: {len(eval_dataset)}, "
          f"Test: {len(test_dataset)}")
    print()

    # ---- Model ----
    model = build_model(
        args.method, tokenizer.vocab_size, d_model, n_heads, n_layers, d_ff,
        args.max_seq_len, **mkw
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    attn_params = sum(
        p.numel() for layer in model.layers
        for p in layer.attn.parameters()
    )
    print(f"Parameters: {total_params/1e6:.1f}M total, "
          f"{attn_params/1e6:.1f}M attention")
    print()

    # ---- Optimizer + Scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- Train ----
    print(f"Training ({total_steps} steps, {args.epochs} epochs)...")
    print("-" * 70)

    best_val_ppl = float('inf')
    best_epoch = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_ppl, tok_s = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            args.grad_clip, args.grad_accum_steps
        )
        val_loss, val_ppl = evaluate(model, eval_loader, device)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch}/{args.epochs}: train_ppl={train_ppl:.2f} "
              f"val_ppl={val_ppl:.2f} tok/s={tok_s:.0f} lr={lr:.6f} "
              f"({elapsed:.0f}s)")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

    print("-" * 70)
    print(f"Best val PPL: {best_val_ppl:.2f} (epoch {best_epoch})")

    # ---- Test ----
    model.load_state_dict(best_state)
    model.to(device)
    test_loss, test_ppl = evaluate(model, test_loader, device)
    print(f"Test PPL: {test_ppl:.2f}")

    # ---- Save ----
    os.makedirs(args.save_dir, exist_ok=True)
    results = {
        'run_name': run_name,
        'experiment': 'D',
        'method': args.method,
        'method_config': mkw,
        'model_size': args.size,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'kv_cache_budget': kv,
        'kv_cache_baseline': kv_base,
        'kv_cache_saving': f"{kv_pct:.1%}",
        'total_params': total_params,
        'attn_params': attn_params,
        'best_val_ppl': round(best_val_ppl, 2),
        'best_epoch': best_epoch,
        'test_ppl': round(test_ppl, 2),
        'test_loss': round(test_loss, 4),
        'training': {
            'epochs': args.epochs,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'seq_len': args.max_seq_len,
            'seed': args.seed,
        },
    }

    save_path = os.path.join(args.save_dir, f'{run_name}.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {save_path}")

    print(f"\nSUMMARY: {args.method:10s} kv={kv:5d} params={total_params/1e6:.1f}M "
          f"val_ppl={best_val_ppl:.2f} test_ppl={test_ppl:.2f}")


if __name__ == '__main__':
    main()

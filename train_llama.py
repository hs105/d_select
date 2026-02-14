"""
Train LLaMA-style model with Asymmetric Attention
===================================================
Trains a small LLaMA (125M params) on WikiText-103.
Compares d_select values on a modern architecture.

Usage:
    python train_llama.py --d_select 192 --size 125M
    python train_llama.py --d_select 768 --size 125M   (baseline)
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from llama_model import AsymmetricLlama


# ============================================================
# Data loading (reused from train.py)
# ============================================================
def load_wikitext(data_path):
    for dirname in ['wikitext-103', 'wikitext-2']:
        wt_dir = os.path.join(data_path, dirname)
        if not os.path.isdir(wt_dir):
            continue
        splits = {}
        for split_name, file_name in [('train', 'wiki.train.tokens'),
                                       ('valid', 'wiki.valid.tokens'),
                                       ('test', 'wiki.test.tokens')]:
            fpath = os.path.join(wt_dir, file_name)
            if os.path.exists(fpath):
                with open(fpath, 'r', encoding='utf-8') as f:
                    text = f.read()
                splits[split_name] = text
                n_words = len(text.split())
                print(f"  {file_name}: {n_words:,} words")
        if 'train' in splits:
            print(f"  Using {dirname}")
            return splits
    return None


class SimpleTokenizer:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

    def build_vocab(self, texts):
        special = ['<pad>', '<unk>', '<eos>']
        for i, tok in enumerate(special):
            self.word2idx[tok] = i
            self.idx2word[i] = tok
        freq = {}
        for text in texts:
            for w in text.lower().split():
                freq[w] = freq.get(w, 0) + 1
        idx = len(special)
        for w, c in sorted(freq.items()):
            if c >= self.min_freq:
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                idx += 1
        self.vocab_size = idx
        print(f"Vocabulary: {self.vocab_size:,} tokens "
              f"(min_freq={self.min_freq}, {len(freq):,} unique words)")

    def encode(self, text):
        unk_id = self.word2idx['<unk>']
        return [self.word2idx.get(w, unk_id) for w in text.lower().split()]

    def encode_texts(self, texts):
        eos_id = self.word2idx['<eos>']
        tokens = []
        for text in texts:
            tokens.extend(self.encode(text))
            tokens.append(eos_id)
        return tokens

    def decode(self, token_ids):
        return ' '.join([self.idx2word.get(t, '<unk>') for t in token_ids])

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'word2idx': self.word2idx, 'min_freq': self.min_freq}, f)


class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.seq_len = seq_len
        self.tokens = tokens
        self.n_sequences = max(0, (len(tokens) - 1) // seq_len)

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ============================================================
# Model configs
# ============================================================
MODEL_CONFIGS = {
    '25M': dict(d_model=512,  n_heads=8,  n_layers=8,  d_ff=1376),
    '50M': dict(d_model=512,  n_heads=8,  n_layers=16, d_ff=1376),
    '125M': dict(d_model=768,  n_heads=12, n_layers=12, d_ff=2048),
    '350M': dict(d_model=1024, n_heads=16, n_layers=24, d_ff=2816),
}


# ============================================================
# Training
# ============================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0,
                grad_accum_steps=1):
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()

    optimizer.zero_grad()
    for step, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)

        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps * x.shape[0] * x.shape[1]
        total_tokens += x.shape[0] * x.shape[1]

    elapsed = time.time() - start_time
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    tok_per_sec = total_tokens / max(elapsed, 1e-6)

    return avg_loss, ppl, tok_per_sec


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_tokens = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)
        total_loss += loss.item() * x.shape[0] * x.shape[1]
        total_tokens += x.shape[0] * x.shape[1]
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    return avg_loss, ppl


@torch.no_grad()
def leak_test(model, vocab_size, seq_len, device, num_batches=10, batch_size=16):
    model.eval()
    total_loss = 0
    total_tokens = 0
    expected = math.log(vocab_size)
    for _ in range(num_batches):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        _, loss = model(x, targets=y)
        total_loss += loss.item() * batch_size * seq_len
        total_tokens += batch_size * seq_len
    actual = total_loss / total_tokens
    print(f"  Leak test: loss={actual:.4f}, expected≈{expected:.4f}")
    if actual < expected * 0.9:
        print("  WARNING: possible leak!")
    else:
        print("  ✓ No leak detected")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Train LLaMA with Asymmetric Attention')

    # Model
    parser.add_argument('--size', type=str, default='125M',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Model size preset')
    parser.add_argument('--d_select', type=int, default=None,
                        help='QK dimension. None=d_model (standard)')
    parser.add_argument('--max_seq_len', type=int, default=512)

    # Data
    parser.add_argument('--data_path', type=str, default='/root/data')
    parser.add_argument('--min_freq', type=int, default=200)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help='Gradient accumulation steps (effective batch = batch_size * accum)')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'none'])

    # Other
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints_llama')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--run_leak_test', action='store_true')
    parser.add_argument('--generate_samples', action='store_true')
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()

    # Get model config
    cfg = MODEL_CONFIGS[args.size]
    d_model = cfg['d_model']
    n_heads = cfg['n_heads']
    n_layers = cfg['n_layers']
    d_ff = cfg['d_ff']

    if args.d_select is None:
        args.d_select = d_model

    if args.run_name is None:
        args.run_name = f"llama_{args.size}_ds{args.d_select}"

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 70)
    print(f"LLaMA Asymmetric Attention — {args.run_name}")
    print("=" * 70)
    print(f"Model:    {args.size} — d_model={d_model}, d_select={args.d_select}, "
          f"n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}")
    print(f"          d_select/head={args.d_select//n_heads}, "
          f"d_value/head={d_model//n_heads}")
    print(f"Training: lr={args.lr}, epochs={args.epochs}, "
          f"batch={args.batch_size}×{args.grad_accum_steps}")
    print(f"Data:     {args.data_path}, seq_len={args.max_seq_len}")
    print(f"Device:   {device}")
    print()

    # ---- Data ----
    print("Loading data...")
    wt = load_wikitext(args.data_path)
    if wt is None:
        print("ERROR: No WikiText data found!")
        return

    train_texts = [line.strip() for line in wt['train'].split('\n')
                   if line.strip() and not line.strip().startswith('=')]
    eval_texts = [line.strip() for line in wt['valid'].split('\n')
                  if line.strip() and not line.strip().startswith('=')]
    test_texts = [line.strip() for line in wt['test'].split('\n')
                  if line.strip() and not line.strip().startswith('=')]

    print(f"  Train: {len(train_texts)} segments")
    print(f"  Valid: {len(eval_texts)} segments")
    print(f"  Test:  {len(test_texts)} segments")
    print()

    tokenizer = SimpleTokenizer(min_freq=args.min_freq)
    tokenizer.build_vocab(train_texts)

    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.save_dir, f'{args.run_name}_tokenizer.json'))

    train_tokens = tokenizer.encode_texts(train_texts)
    eval_tokens = tokenizer.encode_texts(eval_texts)
    test_tokens = tokenizer.encode_texts(test_texts)

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Eval tokens:  {len(eval_tokens):,}")
    print(f"Test tokens:  {len(test_tokens):,}")

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

    print(f"Train sequences: {len(train_dataset)} (seq_len={args.max_seq_len})")
    print(f"Eval sequences:  {len(eval_dataset)}")
    print(f"Test sequences:  {len(test_dataset)}")
    print()

    # ---- Model ----
    model = AsymmetricLlama(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        d_select=args.d_select,
        max_seq_len=args.max_seq_len,
        tie_weights=True,
    ).to(device)

    params = model.count_parameters()
    total_m = params['total'] / 1e6
    print(f"Parameters ({total_m:.1f}M):")
    print(f"  Total: {params['total']:>12,}")
    print(f"  QK:    {params['qk']:>12,}")
    print(f"  VO:    {params['vo']:>12,}")
    print(f"  FFN:   {params['ffn']:>12,}")
    print(f"  Other: {params['other']:>12,} (embeddings, norms)")
    print()

    # ---- Leak test ----
    if args.run_leak_test:
        print("Causal leak test (untrained)...")
        leak_test(model, tokenizer.vocab_size, args.max_seq_len, device)
        print()

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=args.weight_decay
    )

    # ---- Scheduler ----
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    if args.scheduler == 'cosine':
        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # ---- Training ----
    print(f"Training ({total_steps} steps, {args.epochs} epochs)...")
    print("-" * 80)
    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Train PPL':>10} {'tok/s':>8} | "
          f"{'Val Loss':>10} {'Val PPL':>10} | {'LR':>10}")
    print("-" * 80)

    best_val_ppl = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl, tok_per_sec = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            args.grad_clip, args.grad_accum_steps
        )
        val_loss, val_ppl = evaluate(model, eval_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:5d} | {train_loss:10.4f} {train_ppl:10.2f} {tok_per_sec:>7.0f} | "
              f"{val_loss:10.4f} {val_ppl:10.2f} | {current_lr:10.6f}")

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_ppl': val_ppl,
                'args': vars(args),
            }, os.path.join(args.save_dir, f'{args.run_name}_best.pt'))

    print("-" * 80)
    print(f"Best val PPL: {best_val_ppl:.2f} at epoch {best_epoch}")

    # ---- Test ----
    print()
    print("Loading best model for test...")
    ckpt = torch.load(os.path.join(args.save_dir, f'{args.run_name}_best.pt'),
                       map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_ppl = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}")

    # ---- Leak test ----
    if args.run_leak_test:
        print()
        print("Causal leak test (trained)...")
        leak_test(model, tokenizer.vocab_size, args.max_seq_len, device)

    # ---- Generate ----
    if args.generate_samples:
        print()
        print("Generating samples...")
        model.eval()
        for prompt in ["the", "in the", "he was"]:
            toks = tokenizer.encode(prompt)
            ids = torch.tensor([toks], dtype=torch.long, device=device)
            out = model.generate(ids, max_new_tokens=30, temperature=0.8, top_k=40)
            print(f"  '{prompt}' → {tokenizer.decode(out[0].tolist())}")

    # ---- Save results ----
    results = {
        'run_name': args.run_name,
        'size': args.size,
        'd_model': d_model,
        'd_select': args.d_select,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'd_ff': d_ff,
        'params': params,
        'best_val_ppl': best_val_ppl,
        'best_epoch': best_epoch,
        'test_ppl': test_ppl,
        'test_loss': test_loss,
        'vocab_size': tokenizer.vocab_size,
        'args': vars(args),
    }
    rpath = os.path.join(args.save_dir, f'{args.run_name}_results.json')
    with open(rpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {rpath}")


if __name__ == '__main__':
    main()
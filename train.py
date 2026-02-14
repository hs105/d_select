"""
Training Script — Asymmetric Attention on Custom Data
======================================================
Trains a causal language model on text extracted from your dataset.

Supports:
  - JSON QA format (train_data.json / test_data.json): extracts all sentences
  - Plain text files (100sentences.txt, one_sentence.txt)

Usage:
    python train.py --data_path /root/data --d_select 32
    python train.py --data_path /root/data --d_select 256   (baseline)
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from asymmetric_transformer import AsymmetricTransformer


# ============================================================
# Data loading — handles JSON QA format + plain text
# ============================================================
def load_json_qa(path):
    """
    Load QA JSON file. Extract all sentences as training text.
    Format: [{"sentences": ["...", "...", ...], "question": "...", ...}, ...]
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = []
    for item in data:
        if 'question' in item:
            texts.append(item['question'])
        if 'sentences' in item:
            for sent in item['sentences']:
                texts.append(sent)
        if 'answer' in item:
            texts.append(item['answer'])

    return texts


def load_plain_text(path):
    """Load plain text file, split into non-empty lines."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return lines


def load_wikitext(data_path):
    """
    Load WikiText pre-tokenized files.
    Prefers wikitext-103 over wikitext-2 if both exist.
    Returns: dict with 'train', 'valid', 'test' as raw text strings, or None.
    """
    # Prefer wikitext-103 (larger)
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
            else:
                print(f"  WARNING: {fpath} not found")

        if 'train' in splits:
            print(f"  Using {dirname}")
            return splits

    return None


def load_all_text(data_path, source='auto'):
    """
    Load text data.
    source: 'auto' (prefer wikitext if available), 'wikitext', 'json'
    Returns: train_texts, eval_texts, test_texts (lists of strings)
    """
    train_texts = []
    eval_texts = []
    test_texts = []

    # Try WikiText first
    if source in ('auto', 'wikitext'):
        wt = load_wikitext(data_path)
        if wt is not None:
            print("  Using WikiText-2 data")
            # Split into lines, filter empty/header lines
            for line in wt['train'].split('\n'):
                line = line.strip()
                if line and not line.startswith('='):
                    train_texts.append(line)
            if 'valid' in wt:
                for line in wt['valid'].split('\n'):
                    line = line.strip()
                    if line and not line.startswith('='):
                        eval_texts.append(line)
            if 'test' in wt:
                for line in wt['test'].split('\n'):
                    line = line.strip()
                    if line and not line.startswith('='):
                        test_texts.append(line)
            return train_texts, eval_texts, test_texts

    # Fall back to JSON QA format
    if source in ('auto', 'json'):
        train_json = os.path.join(data_path, 'train_data.json')
        if os.path.exists(train_json):
            texts = load_json_qa(train_json)
            print(f"  train_data.json: {len(texts)} text segments")
            train_texts.extend(texts)

        sentences_file = os.path.join(data_path, '100sentences.txt')
        if os.path.exists(sentences_file):
            texts = load_plain_text(sentences_file)
            print(f"  100sentences.txt: {len(texts)} lines")
            train_texts.extend(texts)

        one_sent_file = os.path.join(data_path, 'one_sentence.txt')
        if os.path.exists(one_sent_file):
            texts = load_plain_text(one_sent_file)
            print(f"  one_sentence.txt: {len(texts)} lines")
            train_texts.extend(texts)

        test_json = os.path.join(data_path, 'test_data.json')
        if os.path.exists(test_json):
            texts = load_json_qa(test_json)
            print(f"  test_data.json: {len(texts)} text segments")
            eval_texts.extend(texts)

    # If no separate eval data, split train 90/10
    if not eval_texts and train_texts:
        split = int(len(train_texts) * 0.9)
        eval_texts = train_texts[split:]
        train_texts = train_texts[:split]
        print(f"  No eval file found, split train 90/10")

    return train_texts, eval_texts, test_texts


# ============================================================
# Tokenizer
# ============================================================
class SimpleTokenizer:
    """Word-level tokenizer."""

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

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.min_freq = data['min_freq']
        self.vocab_size = len(self.word2idx)


# ============================================================
# Dataset
# ============================================================
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
# Training
# ============================================================
def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip=1.0):
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, targets=y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * x.shape[0] * x.shape[1]
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
def leak_test(model, vocab_size, seq_len, device, num_batches=10, batch_size=32):
    model.eval()
    total_loss = 0
    total_tokens = 0
    expected_loss = math.log(vocab_size)

    for _ in range(num_batches):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        logits, loss = model(x, targets=y)
        total_loss += loss.item() * batch_size * seq_len
        total_tokens += batch_size * seq_len

    actual_loss = total_loss / total_tokens
    print(f"  Leak test: loss={actual_loss:.4f}, expected≈{expected_loss:.4f}")
    if actual_loss < expected_loss * 0.9:
        print("  WARNING: possible leak!")
    else:
        print("  ✓ No leak detected")
    return actual_loss


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Train Asymmetric Attention LM')

    # Model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--d_select', type=int, default=None,
                        help='QK dimension. None=d_model (standard)')
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--max_seq_len', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--no_tie_weights', action='store_true')

    # Data
    parser.add_argument('--data_path', type=str, default='/root/data')
    parser.add_argument('--source', type=str, default='auto',
                        choices=['auto', 'wikitext', 'json'],
                        help='Data source: auto (prefer wikitext), wikitext, json')
    parser.add_argument('--min_freq', type=int, default=2)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'linear', 'none'])

    # Other
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--run_leak_test', action='store_true')
    parser.add_argument('--generate_samples', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    if args.d_select is None:
        args.d_select = args.d_model

    if args.run_name is None:
        args.run_name = f"dmodel{args.d_model}_dselect{args.d_select}_L{args.n_layers}_H{args.n_heads}"

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 70)
    print(f"Asymmetric Attention LM — {args.run_name}")
    print("=" * 70)
    print(f"Model:    d_model={args.d_model}, d_select={args.d_select}, "
          f"n_heads={args.n_heads}, n_layers={args.n_layers}")
    print(f"          d_select/head={args.d_select//args.n_heads}, "
          f"d_value/head={args.d_model//args.n_heads}")
    print(f"Training: {args.optimizer}, lr={args.lr}, epochs={args.epochs}, "
          f"batch={args.batch_size}")
    print(f"Data:     {args.data_path}")
    print(f"Device:   {device}")
    print()

    # ---- Load data ----
    print("Loading data...")
    train_texts, eval_texts, test_texts = load_all_text(args.data_path, source=args.source)
    print(f"  Train: {len(train_texts)} text segments")
    print(f"  Eval:  {len(eval_texts)} text segments")
    print(f"  Test:  {len(test_texts)} text segments")
    print()

    if not train_texts:
        print("ERROR: No training data found!")
        return

    # ---- Tokenizer ----
    tokenizer = SimpleTokenizer(min_freq=args.min_freq)
    tokenizer.build_vocab(train_texts)

    os.makedirs(args.save_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.save_dir, f'{args.run_name}_tokenizer.json'))

    # ---- Tokenize ----
    train_tokens = tokenizer.encode_texts(train_texts)
    eval_tokens = tokenizer.encode_texts(eval_texts)
    test_tokens = tokenizer.encode_texts(test_texts) if test_texts else []
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Eval tokens:  {len(eval_tokens):,}")
    if test_tokens:
        print(f"Test tokens:  {len(test_tokens):,}")
    print()

    # ---- Datasets ----
    train_dataset = TextDataset(train_tokens, args.max_seq_len)
    eval_dataset = TextDataset(eval_tokens, args.max_seq_len)
    test_dataset = TextDataset(test_tokens, args.max_seq_len) if test_tokens else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    ) if test_dataset else None

    print(f"Train sequences: {len(train_dataset)} (seq_len={args.max_seq_len})")
    print(f"Eval sequences:  {len(eval_dataset)}")
    if test_dataset:
        print(f"Test sequences:  {len(test_dataset)}")
    print()

    # ---- Model ----
    model = AsymmetricTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        d_select=args.d_select,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        tie_weights=not args.no_tie_weights,
    ).to(device)

    param_counts = model.count_parameters()
    print(f"Parameters:")
    print(f"  Total: {param_counts['total']:>10,}")
    print(f"  QK:    {param_counts['qk']:>10,}")
    print(f"  VO:    {param_counts['vo']:>10,}")
    print(f"  Other: {param_counts['other']:>10,}")
    print()

    # ---- Leak test ----
    if args.run_leak_test:
        print("Causal leak test (untrained model)...")
        leak_test(model, tokenizer.vocab_size, args.max_seq_len, device)
        print()

    # ---- Optimizer ----
    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                       weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                     momentum=0.9, weight_decay=args.weight_decay)

    # ---- Scheduler ----
    total_steps = len(train_loader) * args.epochs

    if args.scheduler == 'cosine':
        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.scheduler == 'linear':
        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            return max(0.0, 1.0 - progress)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # ---- Training ----
    print("Training...")
    print("-" * 75)
    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Train PPL':>10} {'tok/s':>8} | "
          f"{'Val Loss':>10} {'Val PPL':>10} | {'LR':>10}")
    print("-" * 75)

    best_val_ppl = float('inf')
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl, tok_per_sec = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_clip
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
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ppl': val_ppl,
                'args': vars(args),
            }, os.path.join(args.save_dir, f'{args.run_name}_best.pt'))

    print("-" * 75)
    print(f"Best validation PPL: {best_val_ppl:.2f} at epoch {best_epoch}")

    # ---- Test evaluation ----
    test_ppl = None
    test_loss = None
    if test_loader is not None:
        print()
        print("Loading best model for test evaluation...")
        checkpoint = torch.load(
            os.path.join(args.save_dir, f'{args.run_name}_best.pt'),
            map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        test_loss, test_ppl = evaluate(model, test_loader, device)
        print(f"Test Loss: {test_loss:.4f}, Test PPL: {test_ppl:.2f}")

    # ---- Leak test on trained model ----
    if args.run_leak_test:
        print()
        print("Causal leak test (trained model)...")
        leak_test(model, tokenizer.vocab_size, args.max_seq_len, device)

    # ---- Generate samples ----
    if args.generate_samples:
        print()
        print("Generating samples...")
        model.eval()
        # Load best model if not already loaded for test eval
        if test_loader is None:
            checkpoint = torch.load(
                os.path.join(args.save_dir, f'{args.run_name}_best.pt'),
                map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint['model_state_dict'])

        prompts = ["the", "what is", "technology has"]
        for prompt_text in prompts:
            prompt_tokens = tokenizer.encode(prompt_text)
            if not prompt_tokens:
                continue
            prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            generated = model.generate(prompt_ids, max_new_tokens=30,
                                        temperature=0.8, top_k=40)
            text = tokenizer.decode(generated[0].tolist())
            print(f"  '{prompt_text}' → {text}")
        print()

    # ---- Save results ----
    results = {
        'run_name': args.run_name,
        'd_model': args.d_model,
        'd_select': args.d_select,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'params': param_counts,
        'best_val_ppl': best_val_ppl,
        'best_epoch': best_epoch,
        'test_ppl': test_ppl,
        'test_loss': test_loss,
        'train_tokens': len(train_tokens),
        'eval_tokens': len(eval_tokens),
        'vocab_size': tokenizer.vocab_size,
        'args': vars(args),
    }
    results_path = os.path.join(args.save_dir, f'{args.run_name}_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
"""
Experiment F/F2/F3/G: SVD Compress + Fine-tune on DIVERSE corpus + Downstream Eval
===================================================================================
Same as svd_finetune_and_eval.py but uses C4 (diverse web text), C4+math mix,
or GSM8K chain-of-thought instead of WikiText-103. Tests whether diverse or
domain-matched fine-tuning data recovers more performance.

Experiment G adds per-layer rank allocation: instead of uniform rank across all
layers, distributes a fixed total rank budget non-uniformly based on each layer's
singular value spectrum.

Full pipeline:
  1. Load Mistral-7B
  2. (Exp G) Compute per-layer rank allocation from SVD spectra
  3. SVD compress W_K to target rank (uniform or per-layer)
  4. Fine-tune QK projections on training data
  5. Run downstream evaluation via lm-eval-harness

Usage:
  # Experiment F: C4 only
  CUDA_VISIBLE_DEVICES=0 python svd_finetune_diverse.py --rank 256 --data c4

  # Experiment F2: C4 + math mixed (7M C4 + 3M math)
  CUDA_VISIBLE_DEVICES=0 python svd_finetune_diverse.py --rank 256 --data c4_math
  CUDA_VISIBLE_DEVICES=1 python svd_finetune_diverse.py --rank 512 --data c4_math
  CUDA_VISIBLE_DEVICES=2 python svd_finetune_diverse.py --rank 1024 --data c4_math

  # Experiment F3: GSM8K chain-of-thought (domain-matched FT for math reasoning)
  CUDA_VISIBLE_DEVICES=3 python svd_finetune_diverse.py --rank 1024 --data gsm8k
  CUDA_VISIBLE_DEVICES=4 python svd_finetune_diverse.py --rank 512  --data gsm8k
  CUDA_VISIBLE_DEVICES=5 python svd_finetune_diverse.py --rank 256  --data gsm8k

  # Experiment G: Per-layer rank allocation
  CUDA_VISIBLE_DEVICES=0 python svd_finetune_diverse.py --rank 256 --alloc energy --data gsm8k
  CUDA_VISIBLE_DEVICES=1 python svd_finetune_diverse.py --rank 256 --alloc greedy --data gsm8k
  CUDA_VISIBLE_DEVICES=2 python svd_finetune_diverse.py --rank 512 --alloc energy --data gsm8k

  # Control (no compression, same diverse FT)
  CUDA_VISIBLE_DEVICES=5 python svd_finetune_diverse.py --rank 1024 --data c4
"""

import argparse
import gc
import glob
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Data loading: diverse corpus
# ============================================================
def load_c4_split(tokenizer, split, seq_len=2048, max_tokens=None):
    """Load C4 (English) via HuggingFace datasets streaming."""
    from datasets import load_dataset
    print(f"  Loading C4/en {split} (streaming)...", flush=True)

    if split == 'validation':
        ds = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
    else:
        ds = load_dataset('allenai/c4', 'en', split='train', streaming=True)

    # Collect text until we have enough tokens
    target_tokens = max_tokens if max_tokens else 10_000_000
    # Add 10% buffer for chunking waste
    target_tokens_with_buffer = int(target_tokens * 1.1)

    all_ids = []
    n_collected = 0
    n_docs = 0

    for doc in ds:
        text = doc['text']
        if not text.strip():
            continue
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        all_ids.append(ids)
        n_collected += len(ids)
        n_docs += 1

        if n_docs % 1000 == 0:
            print(f"    {n_docs} docs, {n_collected:,} tokens...", end='\r', flush=True)

        if n_collected >= target_tokens_with_buffer:
            break

    print(f"    Collected {n_collected:,} tokens from {n_docs} docs", flush=True)

    # Concatenate and chunk
    input_ids = torch.cat(all_ids)
    if max_tokens and len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]

    n_chunks = len(input_ids) // seq_len
    input_ids = input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)
    print(f"  {split}: using {n_chunks * seq_len:,} tokens "
          f"({n_chunks} chunks of {seq_len})", flush=True)
    return input_ids


def load_wikitext103_split(tokenizer, split, seq_len=2048, max_tokens=None):
    """Load WikiText-103 split via HuggingFace datasets (for validation)."""
    from datasets import load_dataset
    print(f"  Loading wikitext-103 {split}...", flush=True)
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
    text = '\n'.join([x['text'] for x in ds if x['text'].strip()])

    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'][0]
    total_tokens = len(input_ids)

    if max_tokens and len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]

    n_chunks = len(input_ids) // seq_len
    input_ids = input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)
    print(f"  {split}: {total_tokens:,} total tokens, "
          f"using {n_chunks * seq_len:,} ({n_chunks} chunks of {seq_len})", flush=True)
    return input_ids


def load_math_split(tokenizer, seq_len=2048, max_tokens=3_000_000,
                    math_dir='/sg-pretrain/datasets/mathematics_dataset-v1.0'):
    """Load DeepMind Mathematics Dataset as formatted Q/A text.

    Reads from all difficulty levels (easy/medium/hard), shuffles, and
    formats as 'Question: ...\nAnswer: ...\n\n' for causal LM training.
    """
    print(f"  Loading math dataset from {math_dir}...", flush=True)

    # Collect all txt files across difficulty levels
    txt_files = []
    for difficulty in ['train-easy', 'train-medium', 'train-hard']:
        pattern = os.path.join(math_dir, difficulty, '*.txt')
        txt_files.extend(sorted(glob.glob(pattern)))
    print(f"    Found {len(txt_files)} files across 3 difficulty levels", flush=True)

    # Read Q/A pairs from all files, sampling proportionally
    all_pairs = []
    for fpath in txt_files:
        with open(fpath, 'r') as f:
            lines = f.read().strip().split('\n')
        # Pairs are consecutive lines: question, answer
        pairs = [(lines[i], lines[i+1]) for i in range(0, len(lines) - 1, 2)]
        all_pairs.extend(pairs)

    random.shuffle(all_pairs)
    print(f"    Total Q/A pairs available: {len(all_pairs):,}", flush=True)

    # Format and tokenize until we reach target tokens
    target_with_buffer = int(max_tokens * 1.1)
    all_ids = []
    n_collected = 0
    n_pairs = 0

    for q, a in all_pairs:
        text = f"Question: {q}\nAnswer: {a}\n\n"
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        all_ids.append(ids)
        n_collected += len(ids)
        n_pairs += 1

        if n_pairs % 10000 == 0:
            print(f"    {n_pairs:,} pairs, {n_collected:,} tokens...", end='\r', flush=True)

        if n_collected >= target_with_buffer:
            break

    print(f"    Collected {n_collected:,} tokens from {n_pairs:,} math pairs", flush=True)

    input_ids = torch.cat(all_ids)
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]

    n_chunks = len(input_ids) // seq_len
    input_ids = input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)
    print(f"  math: using {n_chunks * seq_len:,} tokens "
          f"({n_chunks} chunks of {seq_len})", flush=True)
    return input_ids


def load_gsm8k_split(tokenizer, seq_len=2048, max_tokens=None):
    """Load GSM8K training split with chain-of-thought answers."""
    from datasets import load_dataset
    print("  Loading GSM8K train split...", flush=True)
    ds = load_dataset('gsm8k', 'main', split='train')  # 7,473 examples

    # Format each example as "Question: ...\nAnswer: ...\n\n"
    # Preserves <<calc>> annotations and #### final answer
    all_ids = []
    n_collected = 0
    for ex in ds:
        text = f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False)['input_ids'][0]
        all_ids.append(ids)
        n_collected += len(ids)

    print(f"    Collected {n_collected:,} tokens from {len(ds)} examples", flush=True)

    input_ids = torch.cat(all_ids)
    # GSM8K train is ~1.5-2M tokens total — use all of it
    if max_tokens and len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]

    n_chunks = len(input_ids) // seq_len
    input_ids = input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)
    print(f"  gsm8k: using {n_chunks * seq_len:,} tokens "
          f"({n_chunks} chunks of {seq_len})", flush=True)
    return input_ids


def load_c4_math_mixed(tokenizer, seq_len=2048, max_tokens=10_000_000,
                       c4_tokens=7_000_000, math_tokens=3_000_000,
                       math_dir='/sg-pretrain/datasets/mathematics_dataset-v1.0'):
    """Load mixed corpus: C4 (7M tokens) + math (3M tokens), shuffled.

    Returns chunked input_ids with chunks from both sources interleaved.
    """
    print(f"  Loading mixed C4+math ({c4_tokens/1e6:.0f}M + {math_tokens/1e6:.0f}M tokens)...",
          flush=True)

    c4_ids = load_c4_split(tokenizer, 'train', seq_len, c4_tokens)
    math_ids = load_math_split(tokenizer, seq_len, math_tokens, math_dir)

    print(f"  Combining: {c4_ids.shape[0]} C4 chunks + {math_ids.shape[0]} math chunks",
          flush=True)

    # Concatenate and shuffle chunks
    combined = torch.cat([c4_ids, math_ids], dim=0)
    perm = torch.randperm(combined.shape[0])
    combined = combined[perm]

    total_tokens = combined.shape[0] * seq_len
    print(f"  Mixed total: {total_tokens:,} tokens ({combined.shape[0]} chunks)", flush=True)
    return combined


# ============================================================
# Evaluation (PPL)
# ============================================================
@torch.no_grad()
def evaluate_ppl(model, input_ids, device, batch_size=1):
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
# SVD compression of W_K (on GPU for speed)
# ============================================================
def compress_k_layers(model, rank, device='cuda:0', verbose=True, ranks=None):
    """SVD compress W_K in all layers to target rank.

    Args:
        ranks: Optional list of per-layer ranks. If provided, uses ranks[i]
               for each layer instead of the uniform `rank` argument.
    """
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    k_dim = n_kv_heads * d_head

    if ranks is None and rank >= k_dim:
        if verbose:
            print(f"  Rank {rank} >= K dim {k_dim}, skipping compression (control)")
        return []

    if verbose:
        print(f"  K projection: [{k_dim}, {model.config.hidden_size}]")
        if ranks is not None:
            print(f"  Per-layer ranks: min={min(ranks)}, max={max(ranks)}, "
                  f"avg={sum(ranks)/len(ranks):.0f}")
        else:
            print(f"  Target rank: {rank} (of {k_dim}), saving {1 - rank/k_dim:.0%} of K cache")

    errors = []
    for i in range(n_layers):
        layer_rank = ranks[i] if ranks is not None else rank
        if layer_rank >= k_dim:
            errors.append(0.0)
            if verbose and (i == 0 or i == n_layers - 1 or (i + 1) % 8 == 0):
                print(f"    Layer {i:2d}: rank={layer_rank} (no compression)")
            continue
        W_K = model.model.layers[i].self_attn.k_proj.weight.data.float().to(device)
        U, S, Vh = torch.linalg.svd(W_K, full_matrices=False)
        W_K_compressed = (U[:, :layer_rank] * S[:layer_rank]) @ Vh[:layer_rank, :]
        err = torch.norm(W_K - W_K_compressed).item() / torch.norm(W_K).item()
        errors.append(err)
        model.model.layers[i].self_attn.k_proj.weight.data = W_K_compressed.to(
            device=model.model.layers[i].self_attn.k_proj.weight.device,
            dtype=model.model.layers[i].self_attn.k_proj.weight.dtype,
        )
        if verbose and (i == 0 or i == n_layers - 1 or (i + 1) % 8 == 0):
            print(f"    Layer {i:2d}: rank={layer_rank}, K error = {err:.4f}")

    if verbose:
        print(f"    Average K error: {sum(errors)/len(errors):.4f}", flush=True)
    return errors


def compute_layer_ranks(model, avg_rank, strategy='energy', device='cuda:0',
                        min_rank=32, energy_threshold=0.99):
    """Compute per-layer SVD ranks for a given total budget.

    Args:
        model: The model with K projections to analyze.
        avg_rank: Target average rank per layer.
        strategy: 'energy' (scale by 99%-energy rank) or 'greedy' (water-filling).
        device: Device for SVD computation.
        min_rank: Minimum rank for any layer.
        energy_threshold: Energy fraction for the 'energy' strategy (default 0.99).

    Returns:
        List of n_layers integers summing to n_layers * avg_rank.
    """
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    k_dim = n_kv_heads * d_head
    total_budget = n_layers * avg_rank

    if avg_rank >= k_dim:
        return [k_dim] * n_layers  # no compression needed

    # Compute singular values for all layers
    spectra = []
    for i in range(n_layers):
        W_K = model.model.layers[i].self_attn.k_proj.weight.data.float().to(device)
        S = torch.linalg.svdvals(W_K)
        spectra.append(S.cpu())
        if i % 8 == 0:
            print(f"    SVD spectrum layer {i}/{n_layers}...", end='\r', flush=True)
    print(f"    Computed spectra for {n_layers} layers.", flush=True)

    if strategy == 'energy':
        # Find rank needed to capture energy_threshold of Frobenius norm
        natural_ranks = []
        for i, S in enumerate(spectra):
            energy = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
            hits = (energy >= energy_threshold).nonzero(as_tuple=True)[0]
            rank_needed = hits[0].item() + 1 if len(hits) > 0 else len(S)
            natural_ranks.append(max(rank_needed, min_rank))

        # Scale proportionally to hit total budget
        total_natural = sum(natural_ranks)
        scale = total_budget / total_natural
        ranks = [max(min_rank, min(k_dim, int(round(r * scale)))) for r in natural_ranks]

        # Adjust residual to hit exact budget
        diff = total_budget - sum(ranks)
        if diff > 0:
            # Add rank to layers with most headroom (furthest from k_dim)
            headroom = [(k_dim - ranks[i], i) for i in range(n_layers)]
            headroom.sort(reverse=True)
            for _, i in headroom:
                add = min(diff, k_dim - ranks[i])
                ranks[i] += add
                diff -= add
                if diff <= 0:
                    break
        elif diff < 0:
            # Remove rank from layers with most excess (furthest from min_rank)
            excess = [(ranks[i] - min_rank, i) for i in range(n_layers)]
            excess.sort(reverse=True)
            for _, i in excess:
                sub = min(-diff, ranks[i] - min_rank)
                ranks[i] -= sub
                diff += sub
                if diff >= 0:
                    break

    elif strategy == 'greedy':
        # Water-filling: start at min_rank, greedily allocate to worst layer
        ranks = [min_rank] * n_layers
        remaining = total_budget - sum(ranks)
        step = 32

        while remaining >= step:
            # Find layer with highest relative reconstruction error
            errors = []
            for i, S in enumerate(spectra):
                r = ranks[i]
                if r >= k_dim:
                    errors.append(0.0)
                else:
                    err = (S[r:] ** 2).sum().sqrt() / (S ** 2).sum().sqrt()
                    errors.append(err.item())
            worst = max(range(n_layers), key=lambda i: errors[i])
            add = min(step, k_dim - ranks[worst], remaining)
            ranks[worst] += add
            remaining = total_budget - sum(ranks)

        # Distribute any small remainder one-at-a-time
        while remaining > 0:
            errors = []
            for i, S in enumerate(spectra):
                r = ranks[i]
                if r >= k_dim:
                    errors.append(0.0)
                else:
                    err = (S[r:] ** 2).sum().sqrt() / (S ** 2).sum().sqrt()
                    errors.append(err.item())
            worst = max(range(n_layers), key=lambda i: errors[i])
            ranks[worst] += 1
            remaining -= 1

    else:
        raise ValueError(f"Unknown allocation strategy: {strategy}")

    return ranks


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
# Downstream evaluation via lm-eval
# ============================================================
def run_downstream_eval(model, tokenizer, device, tasks, batch_size='auto'):
    """Run lm-eval-harness on the in-memory model."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    print(f"\nWrapping model for lm-eval...", flush=True)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    task_list = [t.strip() for t in tasks.split(',')]
    print(f"Evaluating on: {task_list}", flush=True)
    t0 = time.time()

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        batch_size=batch_size,
    )

    elapsed = time.time() - t0
    print(f"Evaluation completed in {elapsed/60:.1f} minutes", flush=True)

    # Extract results
    summary = {}
    for task_name, task_results in results['results'].items():
        task_summary = {}
        for metric, value in sorted(task_results.items()):
            if 'alias' in metric:
                continue
            if ',' in metric and isinstance(value, (int, float)):
                clean_metric = metric.split(',')[0]
                suffix = metric.split(',', 1)[1]
                display_name = f"{clean_metric}({suffix})" if suffix != 'none' else clean_metric
                print(f"  {task_name:<20} {display_name:<30} {value:>10.4f}")
                task_summary[display_name] = round(value, 4)
        summary[task_name] = task_summary

    return summary, results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/sg-pretrain/models/mistral-7b')
    parser.add_argument('--rank', type=int, default=256,
                        help='SVD rank for K compression (K dim is 1024)')
    parser.add_argument('--alloc', type=str, default='uniform',
                        choices=['uniform', 'energy', 'greedy'],
                        help='Per-layer rank allocation strategy')
    parser.add_argument('--data', type=str, default='c4',
                        choices=['c4', 'wikitext103', 'c4_math', 'gsm8k'],
                        help='Fine-tuning data source')
    parser.add_argument('--math_dir', type=str,
                        default='/sg-pretrain/datasets/mathematics_dataset-v1.0',
                        help='Path to DeepMind Mathematics Dataset')
    parser.add_argument('--c4_tokens', type=int, default=7_000_000,
                        help='C4 tokens in mixed mode (default 7M)')
    parser.add_argument('--math_tokens', type=int, default=3_000_000,
                        help='Math tokens in mixed mode (default 3M)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--max_train_tokens', type=int, default=10_000_000)
    parser.add_argument('--max_val_tokens', type=int, default=500_000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tasks', type=str,
                        default='mmlu,hellaswag,arc_challenge,winogrande,gsm8k')
    parser.add_argument('--save_dir', type=str,
                        default='/sg-pretrain/focus/paper/experiments/logs')
    args = parser.parse_args()

    device = torch.device(args.device)
    k_dim = 1024  # Mistral-7B: 8 KV heads * 128 dims
    is_compressed = args.rank < k_dim
    data_tag = args.data.replace('wikitext103', 'wt103')
    alloc_tag = f"_{args.alloc}" if args.alloc != 'uniform' else ""
    tag = f"r{args.rank}{alloc_tag}_{data_tag}_ft" if is_compressed else f"control_{data_tag}_ft"
    if args.alloc != 'uniform':
        exp_label = "G"
    elif args.data == 'c4_math':
        exp_label = "F2"
    elif args.data == 'gsm8k':
        exp_label = "F3"
    else:
        exp_label = "F"

    print("=" * 70)
    print(f"Experiment {exp_label}: SVD Compress (rank={args.rank}) + Fine-tune on {args.data}")
    print("=" * 70)

    # Load tokenizer
    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Load data
    print(f"\nLoading training data ({args.data})...", flush=True)
    if args.data == 'c4_math':
        train_ids = load_c4_math_mixed(
            tokenizer, args.seq_len, args.max_train_tokens,
            c4_tokens=args.c4_tokens, math_tokens=args.math_tokens,
            math_dir=args.math_dir)
    elif args.data == 'c4':
        train_ids = load_c4_split(
            tokenizer, 'train', args.seq_len, args.max_train_tokens)
    elif args.data == 'gsm8k':
        train_ids = load_gsm8k_split(
            tokenizer, args.seq_len, args.max_train_tokens)
    else:
        train_ids = load_wikitext103_split(
            tokenizer, 'train', args.seq_len, args.max_train_tokens)

    # Always use WikiText-103 validation for comparable PPL measurement
    print("\nLoading validation data (WikiText-103 for comparable PPL)...", flush=True)
    val_ids = load_wikitext103_split(
        tokenizer, 'validation', args.seq_len, args.max_val_tokens)

    # Load model
    print("\nLoading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_total/1e9:.1f}B params on {device}")

    # Baseline PPL
    print("\nBaseline PPL (before any modification)...", flush=True)
    _, base_ppl = evaluate_ppl(model, val_ids, device)
    print(f"  Val PPL: {base_ppl:.2f}", flush=True)

    # SVD compress
    if is_compressed:
        layer_ranks = None
        if args.alloc != 'uniform':
            print(f"\nComputing per-layer ranks (strategy={args.alloc}, avg_rank={args.rank})...",
                  flush=True)
            layer_ranks = compute_layer_ranks(
                model, args.rank, args.alloc, str(device))
            print(f"  Ranks: {layer_ranks}")
            print(f"  Range: [{min(layer_ranks)}, {max(layer_ranks)}], "
                  f"sum={sum(layer_ranks)}")

        print(f"\nApplying SVD compression (rank={args.rank}) on GPU...", flush=True)
        compress_k_layers(model, args.rank, device=str(device), ranks=layer_ranks)
        torch.cuda.empty_cache()

        _, compressed_ppl = evaluate_ppl(model, val_ids, device)
        delta = (compressed_ppl - base_ppl) / base_ppl * 100
        print(f"  After compression: Val PPL = {compressed_ppl:.2f} ({delta:+.1f}%)", flush=True)
    else:
        compressed_ppl = base_ppl
        print("\nNo compression (control run)", flush=True)

    # Set up optimizer -- QK projections only
    params = []
    for layer in model.model.layers:
        params.append({'params': list(layer.self_attn.q_proj.parameters())})
        params.append({'params': list(layer.self_attn.k_proj.parameters())})

    n_ft = sum(p.numel() for pg in params for p in pg['params'])
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
    print(f"\nFine-tuning for {args.epochs} epochs on {args.data}...", flush=True)
    print("-" * 70)

    best_val_ppl = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss, train_ppl, tok_s = finetune_epoch(
            model, train_ids, optimizer, scheduler, device,
            args.batch_size, args.grad_accum, args.grad_clip
        )
        _, val_ppl = evaluate_ppl(model, val_ids, device)
        delta_val = (val_ppl - base_ppl) / base_ppl * 100

        print(f"  Epoch {epoch}/{args.epochs}: train_ppl={train_ppl:.2f} "
              f"val_ppl={val_ppl:.2f} ({delta_val:+.1f}%) "
              f"tok/s={tok_s:.0f}", flush=True)

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl

    print("-" * 70)

    # Free optimizer memory before eval
    del optimizer, scheduler, train_ids, val_ids
    gc.collect()
    torch.cuda.empty_cache()

    # Downstream eval
    print(f"\n{'='*70}")
    print("DOWNSTREAM EVALUATION")
    print(f"{'='*70}")

    downstream_summary, full_results = run_downstream_eval(
        model, tokenizer, str(device), args.tasks
    )

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    summary = {
        'model': 'mistral-7b',
        'rank': args.rank,
        'alloc': args.alloc,
        'k_dim': k_dim,
        'is_compressed': is_compressed,
        'k_cache_saved': f"{1-args.rank/k_dim:.0%}" if is_compressed else "0%",
        'fine_tune_data': args.data,
        'base_ppl': float(base_ppl),
        'compressed_ppl': float(compressed_ppl),
        'finetuned_best_val_ppl': float(best_val_ppl),
        'epochs': args.epochs,
        'lr': args.lr,
        'max_train_tokens': args.max_train_tokens,
        'tasks': downstream_summary,
    }
    if args.alloc != 'uniform' and is_compressed:
        summary['layer_ranks'] = layer_ranks

    save_path = os.path.join(args.save_dir, f'downstream_{tag}.json')
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {save_path}")

    # Save full lm-eval results
    full_path = os.path.join(args.save_dir, f'downstream_{tag}_full.json')
    serializable = {k: v for k, v in full_results.items() if k in ['results', 'configs']}
    with open(full_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Saved full results to {full_path}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

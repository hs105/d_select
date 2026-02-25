"""
Per-Layer d_select Allocation
==============================
Instead of uniform rank across all layers, allocate a fixed total budget
of rank dimensions across layers based on how compressible each layer is.

Strategy:
1. Compute SVD of W_K per layer, analyze singular value spectra
2. Allocate more rank to layers that "need" it (slower spectral decay)
3. Compare uniform vs allocated compression on GPT-2
"""

import argparse
import json
import math
import os

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_wikitext2_test(tokenizer, seq_len=1024):
    """Load WikiText-2 test set."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
    text = "\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'][0]
    n_chunks = len(input_ids) // seq_len
    input_ids = input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)
    return input_ids


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
    return math.exp(min(avg_loss, 20))


def get_layer_svd_info(model):
    """Compute SVD of W_K for each layer, return singular values and info."""
    n_layers = model.config.n_layer
    d_model = model.config.n_embd
    layer_info = []

    for i in range(n_layers):
        w = model.transformer.h[i].attn.c_attn.weight.data
        W_K = w[:, d_model:2*d_model].float().cpu()

        U, S, Vh = torch.linalg.svd(W_K, full_matrices=False)
        S_np = S.numpy()

        # Cumulative energy (fraction of total Frobenius norm captured)
        total_energy = (S_np ** 2).sum()
        cum_energy = np.cumsum(S_np ** 2) / total_energy

        # Find rank needed for 90%, 95%, 99% energy
        r90 = np.searchsorted(cum_energy, 0.90) + 1
        r95 = np.searchsorted(cum_energy, 0.95) + 1
        r99 = np.searchsorted(cum_energy, 0.99) + 1

        # Effective rank (entropy-based)
        p = (S_np ** 2) / total_energy
        p = p[p > 1e-10]  # avoid log(0)
        eff_rank = np.exp(-np.sum(p * np.log(p)))

        layer_info.append({
            'layer': i,
            'singular_values': S_np,
            'cum_energy': cum_energy,
            'r90': int(r90),
            'r95': int(r95),
            'r99': int(r99),
            'eff_rank': float(eff_rank),
            'top_sv': float(S_np[0]),
            'sv_ratio': float(S_np[0] / S_np[-1]) if S_np[-1] > 0 else float('inf'),
        })

    return layer_info


def allocate_ranks(layer_info, total_budget, strategy='energy95', min_rank=32):
    """Allocate rank budget across layers using different strategies."""
    n_layers = len(layer_info)
    uniform_rank = total_budget // n_layers

    if strategy == 'uniform':
        return [uniform_rank] * n_layers

    elif strategy == 'energy95':
        # Each layer gets what it needs for 95% energy, then redistribute surplus
        needs = [info['r95'] for info in layer_info]
        # Scale to fit budget
        total_needs = sum(needs)
        ranks = [max(min_rank, int(n * total_budget / total_needs)) for n in needs]
        # Adjust to hit exact budget
        while sum(ranks) > total_budget:
            idx = np.argmax(ranks)
            ranks[idx] -= 1
        while sum(ranks) < total_budget:
            idx = np.argmin(ranks)
            ranks[idx] += 1
        return ranks

    elif strategy == 'energy99':
        needs = [info['r99'] for info in layer_info]
        total_needs = sum(needs)
        ranks = [max(min_rank, int(n * total_budget / total_needs)) for n in needs]
        while sum(ranks) > total_budget:
            idx = np.argmax(ranks)
            ranks[idx] -= 1
        while sum(ranks) < total_budget:
            idx = np.argmin(ranks)
            ranks[idx] += 1
        return ranks

    elif strategy == 'eff_rank':
        # Proportional to effective rank
        eff_ranks = [info['eff_rank'] for info in layer_info]
        total_eff = sum(eff_ranks)
        ranks = [max(min_rank, int(e * total_budget / total_eff)) for e in eff_ranks]
        while sum(ranks) > total_budget:
            idx = np.argmax(ranks)
            ranks[idx] -= 1
        while sum(ranks) < total_budget:
            idx = np.argmin(ranks)
            ranks[idx] += 1
        return ranks

    elif strategy == 'inverse_error':
        # Give more rank to layers with higher reconstruction error at uniform rank
        # (layers that suffer most from compression get more budget)
        errors = []
        for info in layer_info:
            sv = info['singular_values']
            # Error at uniform rank = sum of squared discarded SVs
            err = (sv[uniform_rank:] ** 2).sum() / (sv ** 2).sum()
            errors.append(err)
        total_err = sum(errors)
        if total_err == 0:
            return [uniform_rank] * n_layers
        ranks = [max(min_rank, int(e * total_budget / total_err)) for e in errors]
        while sum(ranks) > total_budget:
            idx = np.argmax(ranks)
            ranks[idx] -= 1
        while sum(ranks) < total_budget:
            idx = np.argmin(ranks)
            ranks[idx] += 1
        return ranks

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def compress_k_per_layer(model, ranks):
    """Compress W_K per layer with different ranks."""
    d_model = model.config.n_embd
    errors = []

    for i, rank in enumerate(ranks):
        w = model.transformer.h[i].attn.c_attn.weight.data
        W_K = w[:, d_model:2*d_model].float()

        U, S, Vh = torch.linalg.svd(W_K, full_matrices=False)
        W_K_compressed = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]

        # Reconstruction error
        err = torch.norm(W_K - W_K_compressed).item() / torch.norm(W_K).item()
        errors.append(err)

        w[:, d_model:2*d_model] = W_K_compressed.to(w.dtype)

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_budget', type=int, default=2304,
                        help='Total rank budget across all layers (default: 192*12=2304)')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("Per-Layer d_select Allocation Experiment")
    print("=" * 70)

    # Load
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    test_ids = load_wikitext2_test(tokenizer)
    print(f"Test: {test_ids.shape[0]} chunks of {test_ids.shape[1]}")

    # Baseline
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    base_ppl = evaluate(model, test_ids, device)
    print(f"\nBaseline PPL: {base_ppl:.2f}")
    del model

    # Analyze SVD spectra
    print("\nAnalyzing per-layer SVD spectra...")
    model_tmp = GPT2LMHeadModel.from_pretrained('gpt2')
    layer_info = get_layer_svd_info(model_tmp)
    del model_tmp

    print(f"\n{'Layer':>5} {'EffRank':>8} {'r90':>5} {'r95':>5} {'r99':>5} {'TopSV':>8} {'SV ratio':>10}")
    print("-" * 55)
    for info in layer_info:
        print(f"{info['layer']:5d} {info['eff_rank']:8.1f} {info['r90']:5d} "
              f"{info['r95']:5d} {info['r99']:5d} {info['top_sv']:8.2f} "
              f"{info['sv_ratio']:10.1f}")

    # Test strategies
    n_layers = len(layer_info)
    uniform_rank = args.total_budget // n_layers
    strategies = ['uniform', 'energy95', 'energy99', 'eff_rank', 'inverse_error']

    print(f"\n{'='*70}")
    print(f"Total budget: {args.total_budget} (uniform = {uniform_rank}/layer)")
    print(f"{'='*70}")

    results = {}

    for strategy in strategies:
        # Fresh model each time
        model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

        ranks = allocate_ranks(layer_info, args.total_budget, strategy)
        errors = compress_k_per_layer(model, ranks)
        ppl = evaluate(model, test_ids, device)
        delta = (ppl - base_ppl) / base_ppl * 100

        avg_err = sum(errors) / len(errors)

        print(f"\n--- {strategy} ---")
        print(f"  Ranks: {ranks}")
        print(f"  Range: [{min(ranks)}, {max(ranks)}], std={np.std(ranks):.1f}")
        print(f"  Avg error: {avg_err:.4f}")
        print(f"  PPL: {ppl:.2f} ({delta:+.1f}%)")

        results[strategy] = {
            'ranks': ranks,
            'ppl': float(ppl),
            'delta_pct': float(delta),
            'avg_error': float(avg_err),
            'errors': [float(e) for e in errors],
        }
        del model

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY (budget={args.total_budget}, uniform={uniform_rank}/layer)")
    print(f"{'='*70}")
    print(f"{'Strategy':<20} {'PPL':>8} {'ΔPPL':>8} {'Rank range':>15}")
    print("-" * 55)
    for strategy in strategies:
        r = results[strategy]
        ranks = r['ranks']
        print(f"{strategy:<20} {r['ppl']:8.2f} {r['delta_pct']:>+7.1f}% "
              f"[{min(ranks):3d}, {max(ranks):3d}]")

    # Save
    os.makedirs('./checkpoints_finetune', exist_ok=True)
    with open(f'./checkpoints_finetune/per_layer_budget{args.total_budget}.json', 'w') as f:
        json.dump({
            'total_budget': args.total_budget,
            'uniform_rank': uniform_rank,
            'base_ppl': base_ppl,
            'results': results,
            'layer_info': [{k: v for k, v in info.items() if k != 'singular_values' and k != 'cum_energy'}
                          for info in layer_info],
        }, f, indent=2)
    print(f"\nSaved results.")


if __name__ == '__main__':
    main()
"""
Post-Training QK Compression via SVD
======================================
Take a pretrained GPT-2, compress W_Q and W_K via low-rank SVD approximation,
keep W_V and W_O untouched, measure perplexity on WikiText-2.

This validates: can you reduce QK dimensionality in an already-trained model
without retraining?

Maps directly to our finding:
  SVD rank r = d_select (our asymmetric attention dimension)

Usage:
    python compress_qk.py                          # sweep all ranks
    python compress_qk.py --rank 64 --model gpt2   # single rank
    python compress_qk.py --model gpt2-medium       # larger model
"""

import argparse
import math
import time
import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def load_model_and_tokenizer(model_name, device):
    """Load pretrained GPT-2 model and tokenizer."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    print(f"Loading {model_name}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    d_model = model.config.n_embd
    n_heads = model.config.n_head
    n_layers = model.config.n_layer
    d_head = d_model // n_heads

    print(f"  d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, d_head={d_head}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    return model, tokenizer


def load_wikitext2(tokenizer, seq_len=1024, data_path=None):
    """
    Load WikiText-2 test set for evaluation.
    Tries local files first, then HuggingFace datasets.
    """
    text = None

    # Try local file
    if data_path:
        local_path = os.path.join(data_path, 'wikitext-2', 'wiki.test.tokens')
        if os.path.exists(local_path):
            print(f"Loading WikiText-2 test from {local_path}")
            with open(local_path, 'r') as f:
                text = f.read()

    # Try HuggingFace datasets
    if text is None:
        try:
            from datasets import load_dataset
            print("Loading WikiText-2 test from HuggingFace...")
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split="test")
            text = "\n".join(ds["text"])
        except Exception as e:
            print(f"HuggingFace failed: {e}")

    if text is None:
        raise RuntimeError("Cannot load WikiText-2 test set")

    # Tokenize
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'][0]
    print(f"  Test tokens: {len(input_ids):,}")

    # Split into chunks
    n_chunks = len(input_ids) // seq_len
    input_ids = input_ids[:n_chunks * seq_len].view(n_chunks, seq_len)
    print(f"  Chunks: {n_chunks} × {seq_len}")

    return input_ids


@torch.no_grad()
def evaluate_perplexity(model, input_ids, device, batch_size=4):
    """Compute perplexity on chunked input_ids."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        n_tokens = batch.numel()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def svd_compress_weight(weight, rank):
    """
    Compress a weight matrix via truncated SVD.

    weight: [out_dim, in_dim]
    Returns: compressed weight [out_dim, in_dim] (same shape, lower rank)

    Also returns the two factors A [out_dim, rank] and B [rank, in_dim]
    so that weight ≈ A @ B
    """
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)

    # Truncate to rank
    U_r = U[:, :rank]         # [out_dim, rank]
    S_r = S[:rank]             # [rank]
    Vh_r = Vh[:rank, :]        # [rank, in_dim]

    # Reconstruct: A @ B where A = U_r @ diag(S_r), B = Vh_r
    A = U_r * S_r.unsqueeze(0)   # [out_dim, rank]
    B = Vh_r                      # [rank, in_dim]

    compressed = A @ B            # [out_dim, in_dim]

    # Compute approximation error
    error = torch.norm(weight.float() - compressed) / torch.norm(weight.float())

    return compressed.to(weight.dtype), error.item(), A, B


def get_qkv_weights(model, layer_idx):
    """
    Get Q, K, V weight matrices from a GPT-2 layer.

    GPT-2 stores Q, K, V as one concatenated matrix:
      c_attn.weight: [d_model, 3*d_model]  (note: GPT-2 uses Conv1D, so it's transposed)
      c_attn.bias: [3*d_model]

    Split into W_Q, W_K, W_V each of shape [d_model, d_model].
    """
    layer = model.transformer.h[layer_idx]
    # GPT-2 Conv1D stores weight as [in_features, out_features]
    # so c_attn.weight is [d_model, 3*d_model]
    weight = layer.attn.c_attn.weight.data   # [d_model, 3*d_model]
    bias = layer.attn.c_attn.bias.data       # [3*d_model]

    d_model = weight.shape[0]

    W_Q = weight[:, :d_model]           # [d_model, d_model]
    W_K = weight[:, d_model:2*d_model]  # [d_model, d_model]
    W_V = weight[:, 2*d_model:]         # [d_model, d_model]

    b_Q = bias[:d_model]
    b_K = bias[d_model:2*d_model]
    b_V = bias[2*d_model:]

    return W_Q, W_K, W_V, b_Q, b_K, b_V


def set_qkv_weights(model, layer_idx, W_Q, W_K, W_V, b_Q=None, b_K=None, b_V=None):
    """Write Q, K, V weights back into the GPT-2 layer."""
    layer = model.transformer.h[layer_idx]

    weight = layer.attn.c_attn.weight.data
    d_model = weight.shape[0]

    weight[:, :d_model] = W_Q
    weight[:, d_model:2*d_model] = W_K
    weight[:, 2*d_model:] = W_V

    if b_Q is not None:
        bias = layer.attn.c_attn.bias.data
        bias[:d_model] = b_Q
        bias[d_model:2*d_model] = b_K
        bias[2*d_model:] = b_V


def compress_model_qk(model, rank, compress_v=False, verbose=True):
    """
    Compress Q and K projections in all layers via SVD.
    Optionally compress V too (for comparison).

    Returns: dict of compression statistics.
    """
    n_layers = model.config.n_layer
    d_model = model.config.n_embd

    total_q_error = 0
    total_k_error = 0
    total_v_error = 0

    for i in range(n_layers):
        W_Q, W_K, W_V, b_Q, b_K, b_V = get_qkv_weights(model, i)

        # Compress Q
        W_Q_compressed, q_err, _, _ = svd_compress_weight(W_Q, rank)

        # Compress K
        W_K_compressed, k_err, _, _ = svd_compress_weight(W_K, rank)

        total_q_error += q_err
        total_k_error += k_err

        if compress_v:
            W_V_compressed, v_err, _, _ = svd_compress_weight(W_V, rank)
            total_v_error += v_err
        else:
            W_V_compressed = W_V
            v_err = 0.0

        # Write back
        set_qkv_weights(model, i, W_Q_compressed, W_K_compressed, W_V_compressed,
                         b_Q, b_K, b_V)

        if verbose and (i == 0 or i == n_layers - 1):
            print(f"  Layer {i}: Q_err={q_err:.4f}, K_err={k_err:.4f}, V_err={v_err:.4f}")

    stats = {
        'rank': rank,
        'avg_q_error': total_q_error / n_layers,
        'avg_k_error': total_k_error / n_layers,
        'avg_v_error': total_v_error / n_layers if compress_v else 0.0,
        'original_qk_params': 2 * d_model * d_model * n_layers,
        'compressed_qk_params': 2 * (d_model * rank + rank * d_model) * n_layers,
        # Note: we don't actually change the stored format here,
        # just the values. In production you'd store the factors.
    }

    return stats


def analyze_singular_values(model):
    """
    Analyze singular value distribution of Q, K, V across layers.
    Shows how much energy is in top-k singular values.
    """
    n_layers = model.config.n_layer
    d_model = model.config.n_embd

    print("\nSingular value energy analysis:")
    print(f"  Fraction of energy (sum of squared singular values) in top-r dimensions")
    print(f"  {'Layer':>5} | {'r=16':>8} {'r=32':>8} {'r=64':>8} {'r=128':>8} {'r=256':>8} | QvsV")
    print(f"  {'-'*5} | {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} | {'-'*5}")

    ranks_to_check = [16, 32, 64, 128, 256]

    q_energies_all = []
    v_energies_all = []

    for i in range(n_layers):
        W_Q, W_K, W_V, _, _, _ = get_qkv_weights(model, i)

        _, S_Q, _ = torch.linalg.svd(W_Q.float(), full_matrices=False)
        _, S_V, _ = torch.linalg.svd(W_V.float(), full_matrices=False)

        total_energy_q = (S_Q ** 2).sum().item()
        total_energy_v = (S_V ** 2).sum().item()

        q_fracs = []
        v_fracs = []
        for r in ranks_to_check:
            r = min(r, len(S_Q))
            q_frac = (S_Q[:r] ** 2).sum().item() / total_energy_q
            v_frac = (S_V[:r] ** 2).sum().item() / total_energy_v
            q_fracs.append(q_frac)
            v_fracs.append(v_frac)

        q_energies_all.append(q_fracs)
        v_energies_all.append(v_fracs)

        # Print first, middle, and last layers
        if i == 0 or i == n_layers // 2 or i == n_layers - 1:
            frac_strs = " ".join([f"{f:8.1%}" for f in q_fracs])
            # Compare Q vs V at rank 64
            r64_idx = ranks_to_check.index(64) if 64 in ranks_to_check else 2
            q_v_diff = q_fracs[r64_idx] - v_fracs[r64_idx]
            print(f"  Q {i:>3} | {frac_strs} | {'Q>V' if q_v_diff > 0 else 'V>Q'}")
            frac_strs_v = " ".join([f"{f:8.1%}" for f in v_fracs])
            print(f"  V {i:>3} | {frac_strs_v} |")

    # Average across layers
    avg_q = [sum(q[j] for q in q_energies_all) / n_layers for j in range(len(ranks_to_check))]
    avg_v = [sum(v[j] for v in v_energies_all) / n_layers for j in range(len(ranks_to_check))]
    print(f"  {'':>5} | {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} |")
    print(f"  Q avg | {' '.join(f'{f:8.1%}' for f in avg_q)} |")
    print(f"  V avg | {' '.join(f'{f:8.1%}' for f in avg_v)} |")

    return avg_q, avg_v


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Compress QK via SVD in pretrained GPT-2')
    parser.add_argument('--model', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument('--rank', type=int, default=None,
                        help='Single rank to test. None=sweep all.')
    parser.add_argument('--compress_v', action='store_true',
                        help='Also compress V (for comparison)')
    parser.add_argument('--data_path', type=str, default='/root/data',
                        help='Path to local WikiText data')
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--save_dir', type=str, default='./compression_results')
    parser.add_argument('--analyze_svd', action='store_true',
                        help='Analyze singular value distribution before compression')

    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("POST-TRAINING QK COMPRESSION VIA SVD")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print()

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    d_model = model.config.n_embd
    n_heads = model.config.n_head
    n_layers = model.config.n_layer

    # Load eval data
    print()
    eval_ids = load_wikitext2(tokenizer, args.seq_len, args.data_path)
    print()

    # Baseline perplexity (uncompressed)
    print("Evaluating baseline (uncompressed)...")
    base_loss, base_ppl = evaluate_perplexity(model, eval_ids, device, args.batch_size)
    print(f"  Baseline PPL: {base_ppl:.2f} (loss: {base_loss:.4f})")
    print()

    # Analyze singular values
    if args.analyze_svd:
        avg_q, avg_v = analyze_singular_values(model)
        print()
        # Reload model (analysis doesn't modify, but be safe)
        model, tokenizer = load_model_and_tokenizer(args.model, device)

    # Determine ranks to sweep
    if args.rank is not None:
        ranks = [args.rank]
    else:
        # Sweep: powers of 2 up to d_model
        ranks = []
        r = 16
        while r <= d_model:
            ranks.append(r)
            r *= 2
        # Add d_model/6, d_model/3 for finer resolution
        extra = [d_model // 6, d_model // 4, d_model // 3]
        ranks = sorted(set(ranks + [r for r in extra if r > 0]))

    print("=" * 70)
    print("COMPRESSION SWEEP: QK only")
    print("=" * 70)
    print(f"Baseline PPL: {base_ppl:.2f}")
    print(f"d_model={d_model}, ranks to test: {ranks}")
    print()

    results = []
    os.makedirs(args.save_dir, exist_ok=True)

    for rank in ranks:
        if rank > d_model:
            continue

        print(f"--- Rank {rank} (d_select/head = {rank // n_heads}) ---")

        # Reload fresh model each time
        model, _ = load_model_and_tokenizer(args.model, device)

        # Compress
        stats = compress_model_qk(model, rank, compress_v=args.compress_v, verbose=True)

        # Evaluate
        loss, ppl = evaluate_perplexity(model, eval_ids, device, args.batch_size)
        ppl_change = (ppl - base_ppl) / base_ppl * 100

        # QK parameter savings
        original_qk = stats['original_qk_params']
        compressed_qk = stats['compressed_qk_params']
        param_savings = (1 - compressed_qk / original_qk) * 100

        # KV cache savings
        original_kv_per_token = 2 * d_model * n_layers      # K + V per token
        compressed_kv_per_token = (rank + d_model) * n_layers  # compressed K + full V
        cache_savings = (1 - compressed_kv_per_token / original_kv_per_token) * 100

        print(f"  PPL: {ppl:.2f} ({ppl_change:+.1f}%), "
              f"QK params: {param_savings:.0f}% saved, "
              f"KV cache: {cache_savings:.0f}% saved")
        print()

        result = {
            'rank': rank,
            'rank_per_head': rank // n_heads,
            'ppl': ppl,
            'loss': loss,
            'ppl_change_pct': ppl_change,
            'avg_q_error': stats['avg_q_error'],
            'avg_k_error': stats['avg_k_error'],
            'qk_param_savings_pct': param_savings,
            'kv_cache_savings_pct': cache_savings,
        }
        results.append(result)

    # Also compress V for comparison (if requested)
    if args.compress_v and args.rank is None:
        print("=" * 70)
        print("COMPARISON: Compress QKV (all three)")
        print("=" * 70)

        for rank in [d_model // 4, d_model // 2]:
            print(f"--- Rank {rank} (QKV all compressed) ---")
            model, _ = load_model_and_tokenizer(args.model, device)
            stats = compress_model_qk(model, rank, compress_v=True, verbose=True)
            loss, ppl = evaluate_perplexity(model, eval_ids, device, args.batch_size)
            ppl_change = (ppl - base_ppl) / base_ppl * 100
            print(f"  PPL: {ppl:.2f} ({ppl_change:+.1f}%) — QKV all compressed")
            print()

    # Summary table
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Model: {args.model} (d_model={d_model}, {n_heads} heads, {n_layers} layers)")
    print(f"Baseline PPL: {base_ppl:.2f}")
    print()
    print(f"{'Rank':>6} {'r/head':>6} {'PPL':>8} {'ΔPPL':>8} {'QK Save':>8} {'KV Save':>8} {'Q err':>8} {'K err':>8}")
    print("-" * 70)
    print(f"{'full':>6} {d_model//n_heads:>6} {base_ppl:>8.2f} {'—':>8} {'0%':>8} {'0%':>8} {'0':>8} {'0':>8}")

    for r in results:
        print(f"{r['rank']:>6} {r['rank_per_head']:>6} {r['ppl']:>8.2f} "
              f"{r['ppl_change_pct']:>+7.1f}% {r['qk_param_savings_pct']:>7.0f}% "
              f"{r['kv_cache_savings_pct']:>7.0f}% {r['avg_q_error']:>8.4f} {r['avg_k_error']:>8.4f}")

    # Save
    all_results = {
        'model': args.model,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_layers': n_layers,
        'baseline_ppl': base_ppl,
        'results': results,
    }
    results_path = os.path.join(args.save_dir, f'{args.model}_qk_compression.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
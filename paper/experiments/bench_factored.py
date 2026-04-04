"""
Experiment B (v2): Factored Key Inference Benchmark
====================================================
Implements ACTUAL factored key inference for Mistral-7B:
  - Per-head SVD of W_K → thin key projection + expansion matrix
  - Absorption of expansion into W_Q (paper's approach)
  - Thin keys cached in KV cache (real memory savings)
  - Attention computed in thin Q/K space, V stays full-dim

This shows the real memory savings from factored keys at inference time.

Usage:
  CUDA_VISIBLE_DEVICES=0 python bench_factored.py --device cuda:0
  CUDA_VISIBLE_DEVICES=0 python bench_factored.py --device cuda:0 --context_lengths 4096 --batch_sizes 1,4  # quick
"""

import argparse
import gc
import json
import math
import os
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Factored attention setup
# ============================================================

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # [bs, 1, seq, dim]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def factorize_model(model, rank):
    """Replace standard attention with factored key attention.

    Per-head SVD of W_K, absorption of expansion into W_Q.
    After this, k_proj outputs thin keys, q_proj outputs thin queries,
    and the KV cache stores thin keys (real memory savings).
    V projection and output projection are unchanged.
    """
    config = model.config
    n_layers = config.num_hidden_layers
    n_kv_heads = config.num_key_value_heads       # 8
    n_q_heads = config.num_attention_heads         # 32
    d_head = config.hidden_size // n_q_heads       # 128
    d_model = config.hidden_size                   # 4096
    n_q_per_kv = n_q_heads // n_kv_heads           # 4
    r_per_head = rank // n_kv_heads                # thin dim per head

    print(f"  Factorizing: rank={rank}, r_per_head={r_per_head}, "
          f"d_head={d_head}, n_kv_heads={n_kv_heads}")

    for layer_idx in range(n_layers):
        attn = model.model.layers[layer_idx].self_attn

        W_K = attn.k_proj.weight.data  # [1024, 4096]
        W_Q = attn.q_proj.weight.data  # [4096, 4096]
        orig_dtype = W_K.dtype
        orig_device = W_K.device

        thin_k_rows = []
        new_q_rows = []

        for h in range(n_kv_heads):
            # Extract per-head key projection
            w_k_h = W_K[h * d_head:(h + 1) * d_head, :].float()  # [128, 4096]

            # Per-head SVD
            U, S, Vh = torch.linalg.svd(w_k_h, full_matrices=False)
            U_r = U[:, :r_per_head]         # [128, r_per_head]
            S_r = S[:r_per_head]             # [r_per_head]
            Vh_r = Vh[:r_per_head, :]        # [r_per_head, 4096]

            # Thin key projection for this head: [r_per_head, 4096]
            w_k_thin_h = torch.diag(S_r) @ Vh_r
            thin_k_rows.append(w_k_thin_h)

            # Absorb U_r into each Q head in this GQA group
            for q_offset in range(n_q_per_kv):
                q_idx = h * n_q_per_kv + q_offset
                w_q_head = W_Q[q_idx * d_head:(q_idx + 1) * d_head, :].float()  # [128, 4096]
                # Contract: U_r^T @ w_q = [r_per_head, 4096]
                w_q_new = U_r.T @ w_q_head
                new_q_rows.append(w_q_new)

        # Stack new projections
        W_K_thin = torch.cat(thin_k_rows, dim=0)  # [rank, 4096]
        W_Q_new = torch.cat(new_q_rows, dim=0)     # [n_q_heads * r_per_head, 4096]

        # Replace k_proj: now outputs rank dims instead of 1024
        attn.k_proj = nn.Linear(d_model, rank, bias=False,
                                device=orig_device, dtype=orig_dtype)
        attn.k_proj.weight.data = W_K_thin.to(dtype=orig_dtype, device=orig_device)

        # Replace q_proj: now outputs n_q_heads * r_per_head dims
        q_out_dim = n_q_heads * r_per_head
        attn.q_proj = nn.Linear(d_model, q_out_dim, bias=False,
                                device=orig_device, dtype=orig_dtype)
        attn.q_proj.weight.data = W_Q_new.to(dtype=orig_dtype, device=orig_device)

        if layer_idx == 0 or layer_idx == n_layers - 1:
            print(f"    Layer {layer_idx}: k_proj [{rank}, {d_model}], "
                  f"q_proj [{q_out_dim}, {d_model}]")

    # Monkey-patch forward for all attention layers (thin QK, full V)
    _patch_attention_forward(model, r_per_head, d_head, n_q_heads, n_kv_heads)
    print(f"  Done. All attention layers patched for factored inference.")


def _make_attention_forward(attn_module, qk_head_dim, v_head_dim, n_q_heads,
                            n_kv_heads):
    """Create an attention forward fn for given QK/V head dims.

    Used for BOTH baseline (qk_head_dim == v_head_dim == 128) and factored
    (qk_head_dim < v_head_dim) so that the two configs traverse the exact
    same Python / CUDA code-path and any throughput difference is attributable
    solely to the dimension change.
    """
    n_q_per_kv = n_q_heads // n_kv_heads

    def forward(
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # Q projection  [bsz, q_len, n_q_heads * qk_head_dim]
        query_states = attn_module.q_proj(hidden_states)
        query_states = query_states.view(
            bsz, q_len, n_q_heads, qk_head_dim).transpose(1, 2)

        # K projection  [bsz, q_len, n_kv_heads * qk_head_dim]
        key_states = attn_module.k_proj(hidden_states)
        key_states = key_states.view(
            bsz, q_len, n_kv_heads, qk_head_dim).transpose(1, 2)

        # V projection  [bsz, q_len, n_kv_heads * v_head_dim]
        value_states = attn_module.v_proj(hidden_states)
        value_states = value_states.view(
            bsz, q_len, n_kv_heads, v_head_dim).transpose(1, 2)

        # RoPE — slice frequencies to match qk_head_dim
        cos_full, sin_full = position_embeddings
        if qk_head_dim == cos_full.shape[-1]:
            # Full dims — no slicing needed
            cos_qk, sin_qk = cos_full, sin_full
        else:
            half = qk_head_dim // 2
            cos_qk = torch.cat(
                [cos_full[..., :half], cos_full[..., :half]], dim=-1)
            sin_qk = torch.cat(
                [sin_full[..., :half], sin_full[..., :half]], dim=-1)

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos_qk, sin_qk)

        # KV cache update
        if past_key_values is not None:
            cache_kwargs = {
                "sin": sin_qk, "cos": cos_qk,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_values.update(
                key_states, value_states,
                attn_module.layer_idx, cache_kwargs)

        # GQA expansion
        key_states = repeat_kv(key_states, n_q_per_kv)
        value_states = repeat_kv(value_states, n_q_per_kv)

        # Attention
        is_causal = (attention_mask is None and q_len > 1)
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, :, :, :key_states.shape[-2]]

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=1.0 / math.sqrt(qk_head_dim),
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Reshape back to [bsz, q_len, n_q_heads * v_head_dim]
        attn_output = attn_output.reshape(
            bsz, q_len, n_q_heads * v_head_dim)

        # Output projection (unchanged)
        attn_output = attn_module.o_proj(attn_output)
        return attn_output, None

    return forward


def _patch_attention_forward(model, qk_head_dim, v_head_dim, n_q_heads,
                             n_kv_heads):
    """Monkey-patch all attention layers to use the unified forward."""
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.forward = _make_attention_forward(
            attn, qk_head_dim, v_head_dim, n_q_heads, n_kv_heads)


# ============================================================
# Benchmarking functions
# ============================================================

@torch.no_grad()
def bench_prefill(model, input_ids, n_warmup=2, n_runs=5):
    """Benchmark prefill throughput."""
    for _ in range(n_warmup):
        model(input_ids, use_cache=True)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = model(input_ids, use_cache=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    n_tokens = input_ids.numel()
    avg_time = sum(times) / len(times)
    throughput = n_tokens / avg_time

    return {
        'avg_time_s': round(avg_time, 4),
        'throughput_tok_s': round(throughput, 1),
        'peak_memory_gb': round(peak_mem, 2),
        'n_tokens': n_tokens,
    }


@torch.no_grad()
def bench_decode(model, input_ids, n_new_tokens=128, n_warmup=1, n_runs=3):
    """Benchmark autoregressive decode."""
    outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    torch.cuda.synchronize()

    # Warmup decode
    for _ in range(n_warmup):
        kv = past_kv
        tok = next_token
        for _ in range(min(16, n_new_tokens)):
            out = model(tok, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            tok = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(n_runs):
        kv = past_kv
        tok = next_token
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_new_tokens):
            out = model(tok, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            tok = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    batch_size = input_ids.shape[0]
    avg_time = sum(times) / len(times)
    throughput = (n_new_tokens * batch_size) / avg_time

    return {
        'avg_time_s': round(avg_time, 4),
        'throughput_tok_s': round(throughput, 1),
        'peak_memory_gb': round(peak_mem, 2),
        'n_new_tokens': n_new_tokens,
        'batch_size': batch_size,
    }


@torch.no_grad()
def measure_kv_cache_size(model, input_ids):
    """Measure actual KV cache memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values

    kv_bytes = 0
    k_bytes = 0
    v_bytes = 0
    k_shape = None
    v_shape = None
    for layer_idx in range(len(past_kv.layers)):
        layer = past_kv.layers[layer_idx]
        k_tensor = layer.keys
        v_tensor = layer.values
        k_b = k_tensor.nelement() * k_tensor.element_size()
        v_b = v_tensor.nelement() * v_tensor.element_size()
        k_bytes += k_b
        v_bytes += v_b
        kv_bytes += k_b + v_b
        if layer_idx == 0:
            k_shape = list(k_tensor.shape)
            v_shape = list(v_tensor.shape)

    return {
        'kv_cache_mb': round(kv_bytes / 1e6, 2),
        'k_cache_mb': round(k_bytes / 1e6, 2),
        'v_cache_mb': round(v_bytes / 1e6, 2),
        'k_shape': k_shape,
        'v_shape': v_shape,
    }


def find_max_batch_size(model, seq_len, device, max_try=128):
    """Binary search for max batch size that fits in GPU memory."""
    lo, hi = 1, max_try
    best = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            gc.collect()
            torch.cuda.empty_cache()
            dummy = torch.randint(1, 1000, (mid, seq_len), device=device)
            outputs = model(dummy, use_cache=True)
            next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
            model(next_tok, past_key_values=outputs.past_key_values, use_cache=True)
            del outputs, dummy, next_tok
            torch.cuda.synchronize()
            best = mid
            lo = mid + 1
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            gc.collect()
            torch.cuda.empty_cache()
            hi = mid - 1

    return best


# ============================================================
# Main
# ============================================================

def run_config(model_path, config_name, rank, device, tokenizer,
               context_lengths, batch_sizes, n_decode_tokens):
    """Load model, optionally factorize, run benchmarks."""
    print(f"\n{'='*70}")
    print(f"Configuration: {config_name}")
    print(f"{'='*70}")

    print(f"\nLoading model...", flush=True)
    # IMPORTANT: Both baseline and factored use "eager" loading + the same
    # monkey-patched forward (via _make_attention_forward) so that the only
    # variable is the QK dimension.  The old code used "sdpa" for baseline
    # and "eager" for factored, which was an unfair comparison.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    config = model.config
    n_q_heads = config.num_attention_heads          # 32
    n_kv_heads = config.num_key_value_heads         # 8
    d_head = config.hidden_size // n_q_heads        # 128

    if rank is not None:
        factorize_model(model, rank)
        torch.cuda.empty_cache()
    else:
        # Baseline: monkey-patch with full dims so code path is identical
        _patch_attention_forward(model, d_head, d_head, n_q_heads, n_kv_heads)
        print(f"  Baseline patched with same forward as factored (d_head={d_head}).")

    results = {}

    for seq_len in context_lengths:
        print(f"\n--- Context length: {seq_len} ---")

        for bs in batch_sizes:
            config_key = f"ctx{seq_len}_bs{bs}"
            print(f"\n  Batch size {bs}, seq_len {seq_len}:")

            try:
                gc.collect()
                torch.cuda.empty_cache()

                input_ids = torch.randint(
                    1, tokenizer.vocab_size, (bs, seq_len), device=device)

                # KV cache size (at bs=1)
                if bs == 1:
                    kv_info = measure_kv_cache_size(model, input_ids)
                    print(f"    KV cache: {kv_info['kv_cache_mb']:.1f} MB "
                          f"(K={kv_info['k_cache_mb']:.1f}, V={kv_info['v_cache_mb']:.1f})")
                    print(f"    K shape: {kv_info['k_shape']}, V shape: {kv_info['v_shape']}")
                else:
                    kv_info = {}

                # Prefill
                gc.collect()
                torch.cuda.empty_cache()
                prefill = bench_prefill(model, input_ids, n_warmup=2, n_runs=5)
                print(f"    Prefill: {prefill['throughput_tok_s']:.0f} tok/s, "
                      f"TTFT={prefill['avg_time_s']*1000:.1f}ms, "
                      f"peak={prefill['peak_memory_gb']:.1f}GB")

                # Decode
                gc.collect()
                torch.cuda.empty_cache()
                decode = bench_decode(
                    model, input_ids,
                    n_new_tokens=n_decode_tokens,
                    n_warmup=1, n_runs=3)
                print(f"    Decode:  {decode['throughput_tok_s']:.0f} tok/s, "
                      f"peak={decode['peak_memory_gb']:.1f}GB")

                results[config_key] = {
                    'prefill': prefill,
                    'decode': decode,
                    'kv_cache': kv_info,
                    'seq_len': seq_len,
                    'batch_size': bs,
                }

                del input_ids
                gc.collect()
                torch.cuda.empty_cache()

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if 'out of memory' in str(e).lower():
                    print(f"    OOM! Skipping bs={bs} at ctx={seq_len}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    results[config_key] = {'error': 'OOM'}
                    break
                else:
                    raise

        # Max batch size
        print(f"\n  Finding max batch size for ctx={seq_len}...")
        gc.collect()
        torch.cuda.empty_cache()
        max_bs = find_max_batch_size(model, seq_len, device, max_try=64)
        print(f"    Max batch size: {max_bs}")
        results[f"ctx{seq_len}_max_bs"] = max_bs

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/sg-pretrain/models/mistral-7b')
    parser.add_argument('--ranks', type=str, default='256,512',
                        help='Comma-separated ranks for factored configs')
    parser.add_argument('--context_lengths', type=str, default='4096,16384',
                        help='Comma-separated context lengths')
    parser.add_argument('--batch_sizes', type=str, default='1,4,8,16,32',
                        help='Comma-separated batch sizes')
    parser.add_argument('--n_decode_tokens', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str,
                        default='/sg-pretrain/focus/paper/experiments/logs')
    args = parser.parse_args()

    device = torch.device(args.device)
    ranks = [int(r) for r in args.ranks.split(',')]
    context_lengths = [int(c) for c in args.context_lengths.split(',')]
    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]

    print("=" * 70)
    print("Experiment B: Factored Key Inference Benchmark")
    print("=" * 70)
    print(f"  Ranks: {ranks}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    all_results = {}

    # 1. Baseline (same monkey-patched forward, full dims — fair comparison)
    all_results['baseline'] = run_config(
        args.model_path, 'baseline (standard keys)', None, device,
        tokenizer, context_lengths, batch_sizes, args.n_decode_tokens)

    # 2. Factored configs
    for rank in ranks:
        tag = f"factored_r{rank}"
        all_results[tag] = run_config(
            args.model_path, f'factored (rank={rank})', rank, device,
            tokenizer, context_lengths, batch_sizes, args.n_decode_tokens)

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'factored_bench.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {save_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # KV cache comparison
    print(f"\n{'Config':<20} {'Ctx':>6} {'K cache MB':>10} {'V cache MB':>10} "
          f"{'KV total MB':>12} {'K shape':>20}")
    print("-" * 80)
    for tag in all_results:
        for key, val in all_results[tag].items():
            if isinstance(val, dict) and 'kv_cache' in val and val.get('kv_cache'):
                kv = val['kv_cache']
                print(f"{tag:<20} {val['seq_len']:>6} "
                      f"{kv.get('k_cache_mb', 0):>10.1f} "
                      f"{kv.get('v_cache_mb', 0):>10.1f} "
                      f"{kv.get('kv_cache_mb', 0):>12.1f} "
                      f"{str(kv.get('k_shape', '')):>20}")

    # Throughput comparison
    print(f"\n{'Config':<20} {'Ctx':>6} {'BS':>4} {'Prefill tok/s':>14} "
          f"{'Decode tok/s':>13} {'Peak GB':>8}")
    print("-" * 70)
    for tag in all_results:
        for key, val in all_results[tag].items():
            if isinstance(val, dict) and 'prefill' in val:
                print(f"{tag:<20} {val['seq_len']:>6} {val['batch_size']:>4} "
                      f"{val['prefill']['throughput_tok_s']:>14.0f} "
                      f"{val['decode']['throughput_tok_s']:>13.0f} "
                      f"{val['decode']['peak_memory_gb']:>8.1f}")

    # Max batch size
    print(f"\n{'Config':<20} {'Ctx':>6} {'Max BS':>7}")
    print("-" * 35)
    for tag in all_results:
        for key, val in all_results[tag].items():
            if key.endswith('max_bs'):
                ctx = key.replace('ctx', '').replace('_max_bs', '')
                print(f"{tag:<20} {ctx:>6} {val:>7}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

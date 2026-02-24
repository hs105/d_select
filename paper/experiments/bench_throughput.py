"""
Experiment B: End-to-End Throughput Benchmarks for Asymmetric Attention
======================================================================
Measures actual inference throughput, peak memory, and max batch size
for standard vs SVD-compressed Mistral-7B at various context lengths.

Key metrics:
  - Prefill throughput (tokens/sec)
  - Decode throughput (tokens/sec)
  - Peak GPU memory (GB)
  - Max batch size that fits in memory
  - Time-to-first-token (TTFT)

Usage:
  # Full benchmark on GPU 0
  CUDA_VISIBLE_DEVICES=0 python bench_throughput.py --device cuda:0

  # Quick test
  CUDA_VISIBLE_DEVICES=0 python bench_throughput.py --device cuda:0 --context_lengths 4096 --batch_sizes 1,4
"""

import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compress_k_layers(model, rank, device='cuda:0'):
    """SVD compress W_K in all layers to target rank."""
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    k_dim = n_kv_heads * d_head

    if rank >= k_dim:
        print(f"  Rank {rank} >= K dim {k_dim}, skipping (baseline)")
        return

    print(f"  Compressing K to rank {rank} (saving {1-rank/k_dim:.0%})...")
    for i in range(n_layers):
        W_K = model.model.layers[i].self_attn.k_proj.weight.data.float().to(device)
        U, S, Vh = torch.linalg.svd(W_K, full_matrices=False)
        W_K_compressed = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
        model.model.layers[i].self_attn.k_proj.weight.data = W_K_compressed.to(
            device=model.model.layers[i].self_attn.k_proj.weight.device,
            dtype=model.model.layers[i].self_attn.k_proj.weight.dtype,
        )
    print(f"  Done.", flush=True)


@torch.no_grad()
def bench_prefill(model, input_ids, n_warmup=2, n_runs=5):
    """Benchmark prefill (processing the full prompt)."""
    # Warmup
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
        'times': [round(t, 4) for t in times],
    }


@torch.no_grad()
def bench_decode(model, input_ids, n_new_tokens=128, n_warmup=1, n_runs=3):
    """Benchmark autoregressive decode (generating tokens one at a time)."""
    # First do prefill to get past_key_values
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
        'times': [round(t, 4) for t in times],
    }


@torch.no_grad()
def measure_kv_cache_size(model, input_ids):
    """Measure actual KV cache memory by comparing before/after prefill."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mem_before = torch.cuda.memory_allocated()
    outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values
    mem_after = torch.cuda.memory_allocated()

    # Calculate KV cache size
    kv_size_bytes = 0
    for layer_kv in past_kv:
        for tensor in layer_kv:
            kv_size_bytes += tensor.nelement() * tensor.element_size()

    return {
        'kv_cache_gb': round(kv_size_bytes / 1e9, 4),
        'kv_cache_mb': round(kv_size_bytes / 1e6, 2),
        'memory_delta_gb': round((mem_after - mem_before) / 1e9, 4),
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
            # Try a decode step too
            next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
            model(next_tok, past_key_values=outputs.past_key_values, use_cache=True)
            del outputs, dummy, next_tok
            torch.cuda.synchronize()
            best = mid
            lo = mid + 1
        except torch.cuda.OutOfMemoryError:
            del dummy
            gc.collect()
            torch.cuda.empty_cache()
            hi = mid - 1

    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/sg-pretrain/models/mistral-7b')
    parser.add_argument('--ranks', type=str, default='1024,512,256',
                        help='Comma-separated ranks (1024=baseline)')
    parser.add_argument('--context_lengths', type=str, default='4096,16384',
                        help='Comma-separated context lengths')
    parser.add_argument('--batch_sizes', type=str, default='1,4,8,16',
                        help='Comma-separated batch sizes to try')
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
    print("Experiment B: End-to-End Throughput Benchmarks")
    print("=" * 70)
    print(f"  Ranks: {ranks}")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Device: {device}")
    print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    all_results = {}

    for rank in ranks:
        tag = f"r{rank}" if rank < 1024 else "baseline"
        print(f"\n{'='*70}")
        print(f"Configuration: {tag} (rank={rank})")
        print(f"{'='*70}")

        # Load fresh model for each rank
        print(f"\nLoading model...", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to(device)
        model.eval()

        if rank < 1024:
            compress_k_layers(model, rank, device=str(device))
            torch.cuda.empty_cache()

        rank_results = {}

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

                    # KV cache size (batch=1 reference)
                    if bs == 1:
                        kv_info = measure_kv_cache_size(model, input_ids)
                        print(f"    KV cache: {kv_info['kv_cache_mb']:.1f} MB")
                    else:
                        kv_info = {}

                    # Prefill benchmark
                    gc.collect()
                    torch.cuda.empty_cache()
                    prefill = bench_prefill(model, input_ids, n_warmup=2, n_runs=5)
                    print(f"    Prefill: {prefill['throughput_tok_s']:.0f} tok/s, "
                          f"TTFT={prefill['avg_time_s']*1000:.1f}ms, "
                          f"peak={prefill['peak_memory_gb']:.1f}GB")

                    # Decode benchmark
                    gc.collect()
                    torch.cuda.empty_cache()
                    decode = bench_decode(
                        model, input_ids,
                        n_new_tokens=args.n_decode_tokens,
                        n_warmup=1, n_runs=3)
                    print(f"    Decode:  {decode['throughput_tok_s']:.0f} tok/s, "
                          f"peak={decode['peak_memory_gb']:.1f}GB")

                    rank_results[config_key] = {
                        'prefill': prefill,
                        'decode': decode,
                        'kv_cache': kv_info,
                        'seq_len': seq_len,
                        'batch_size': bs,
                    }

                    del input_ids
                    gc.collect()
                    torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError:
                    print(f"    OOM! Skipping bs={bs} at ctx={seq_len}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    rank_results[config_key] = {'error': 'OOM'}
                    # Skip larger batch sizes for this context length
                    break

            # Find max batch size
            print(f"\n  Finding max batch size for ctx={seq_len}...")
            gc.collect()
            torch.cuda.empty_cache()
            max_bs = find_max_batch_size(model, seq_len, device, max_try=64)
            print(f"    Max batch size: {max_bs}")
            rank_results[f"ctx{seq_len}_max_bs"] = max_bs

        all_results[tag] = rank_results

        # Free model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, 'throughput_bench.json')
    with open(save_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {save_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'Ctx':>6} {'BS':>4} {'Prefill tok/s':>14} {'Decode tok/s':>13} {'Peak GB':>8} {'KV MB':>8}")
    print("-" * 70)

    for tag in all_results:
        for key, val in all_results[tag].items():
            if key.startswith('ctx') and not key.endswith('max_bs') and 'error' not in val:
                kv_mb = val.get('kv_cache', {}).get('kv_cache_mb', '')
                kv_str = f"{kv_mb:.1f}" if kv_mb else "--"
                print(f"{tag:<15} {val['seq_len']:>6} {val['batch_size']:>4} "
                      f"{val['prefill']['throughput_tok_s']:>14.0f} "
                      f"{val['decode']['throughput_tok_s']:>13.0f} "
                      f"{val['decode']['peak_memory_gb']:>8.1f} "
                      f"{kv_str:>8}")

    print(f"\n{'Config':<15} {'Ctx':>6} {'Max BS':>7}")
    print("-" * 30)
    for tag in all_results:
        for key, val in all_results[tag].items():
            if key.endswith('max_bs'):
                ctx = key.replace('ctx', '').replace('_max_bs', '')
                print(f"{tag:<15} {ctx:>6} {val:>7}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

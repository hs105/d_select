"""
Experiment A: Downstream Task Evaluation for SVD-Compressed Mistral-7B
======================================================================
Evaluates baseline and SVD-compressed Mistral-7B on standard benchmarks
using lm-evaluation-harness.

Tasks: MMLU (5-shot), Hellaswag (10-shot), ARC-Challenge (25-shot),
       WinoGrande (5-shot), GSM8K (5-shot CoT)

Usage:
  # Baseline (no compression)
  python eval_downstream.py --rank 1024 --device cuda:0

  # Compressed (75% K cache saved)
  python eval_downstream.py --rank 256 --device cuda:4

  # Quick test (just hellaswag)
  python eval_downstream.py --rank 1024 --device cuda:0 --tasks hellaswag --limit 100
"""

import argparse
import json
import math
import os
import time

import torch
import lm_eval
from lm_eval.models.huggingface import HFLM


def compress_k_layers(model, rank, device='cuda:0', verbose=True):
    """SVD compress W_K in all layers to target rank. Uses GPU for fast SVD."""
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    k_dim = n_kv_heads * d_head  # 1024 for Mistral-7B

    if rank >= k_dim:
        if verbose:
            print(f"  Rank {rank} >= K dim {k_dim}, skipping compression (baseline)")
        return []

    if verbose:
        print(f"  K projection: [{k_dim}, {model.config.hidden_size}]")
        print(f"  Target rank: {rank} (of {k_dim}), saving {1 - rank/k_dim:.0%} of K cache")

    svd_device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    errors = []
    for i in range(n_layers):
        W_K = model.model.layers[i].self_attn.k_proj.weight.data.float().to(svd_device)
        U, S, Vh = torch.linalg.svd(W_K, full_matrices=False)
        W_K_compressed = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
        err = torch.norm(W_K - W_K_compressed).item() / torch.norm(W_K).item()
        errors.append(err)
        model.model.layers[i].self_attn.k_proj.weight.data = W_K_compressed.to(
            device=model.model.layers[i].self_attn.k_proj.weight.device,
            dtype=model.model.layers[i].self_attn.k_proj.weight.dtype,
        )
        if verbose and (i == 0 or i == n_layers - 1 or (i + 1) % 8 == 0):
            print(f"    Layer {i:2d}: K error = {err:.4f}")

    if verbose:
        print(f"    Average K error: {sum(errors)/len(errors):.4f}", flush=True)
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/sg-pretrain/models/mistral-7b')
    parser.add_argument('--rank', type=int, default=1024,
                        help='SVD rank for K compression (1024 = no compression)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tasks', type=str,
                        default='mmlu,hellaswag,arc_challenge,winogrande,gsm8k',
                        help='Comma-separated list of tasks')
    parser.add_argument('--batch_size', type=str, default='auto',
                        help='Batch size for evaluation (auto recommended)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of examples per task (for testing)')
    parser.add_argument('--save_dir', type=str,
                        default='/sg-pretrain/focus/paper/experiments/logs')
    args = parser.parse_args()

    print("=" * 70)
    print(f"Downstream Evaluation: Mistral-7B (rank={args.rank})")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from {args.model_path}...", flush=True)
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    # Apply SVD compression if needed
    k_dim = (model.config.num_key_value_heads *
             (model.config.hidden_size // model.config.num_attention_heads))
    is_compressed = args.rank < k_dim

    # Move to device first (so SVD runs on GPU -- 100x faster)
    print(f"\nMoving to {args.device}...", flush=True)
    model = model.to(args.device)

    if is_compressed:
        print(f"\nApplying SVD compression (rank={args.rank}) on GPU...", flush=True)
        errors = compress_k_layers(model, args.rank, device=args.device)
    else:
        print("\nNo compression (baseline run)", flush=True)
        errors = []
    torch.cuda.empty_cache()

    # Wrap in lm-eval HFLM
    print("\nWrapping model for lm-eval...", flush=True)
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Run evaluation
    task_list = [t.strip() for t in args.tasks.split(',')]
    print(f"\nEvaluating on: {task_list}", flush=True)
    print(f"Limit: {args.limit if args.limit else 'None (full)'}", flush=True)
    t0 = time.time()

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    elapsed = time.time() - t0
    print(f"\nEvaluation completed in {elapsed/60:.1f} minutes", flush=True)

    # Extract and print results
    print(f"\n{'='*70}")
    print(f"RESULTS: rank={args.rank} ({'baseline' if not is_compressed else f'{1-args.rank/k_dim:.0%} K cache saved'})")
    print(f"{'='*70}")
    print(f"{'Task':<20} {'Metric':<25} {'Score':>10}")
    print("-" * 55)

    summary = {
        'model': 'mistral-7b',
        'rank': args.rank,
        'k_dim': k_dim,
        'is_compressed': is_compressed,
        'k_cache_saved': f"{1-args.rank/k_dim:.0%}" if is_compressed else "0%",
        'tasks': {},
        'elapsed_minutes': round(elapsed / 60, 1),
    }

    for task_name, task_results in results['results'].items():
        task_summary = {}
        for metric, value in sorted(task_results.items()):
            if 'alias' in metric:
                continue
            # Handle both ",none" and ",strict-match"/",flexible-extract" suffixes
            if ',' in metric and isinstance(value, (int, float)):
                clean_metric = metric.split(',')[0]
                suffix = metric.split(',', 1)[1]
                if 'stderr' in clean_metric:
                    display_name = f"{clean_metric}({suffix})"
                else:
                    display_name = f"{clean_metric}({suffix})" if suffix != 'none' else clean_metric
                print(f"{task_name:<20} {display_name:<30} {value:>10.4f}")
                task_summary[display_name] = round(value, 4)
        summary['tasks'][task_name] = task_summary

    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    tag = f"r{args.rank}" if is_compressed else "baseline"
    save_path = os.path.join(args.save_dir, f'downstream_{tag}.json')
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {save_path}")

    # Also save full lm-eval results
    full_path = os.path.join(args.save_dir, f'downstream_{tag}_full.json')
    # Filter out non-serializable items
    serializable = {k: v for k, v in results.items() if k in ['results', 'configs']}
    with open(full_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Saved full results to {full_path}")


if __name__ == '__main__':
    main()

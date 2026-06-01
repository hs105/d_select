"""
Downstream Task Evaluation for 7B From-Scratch Models (Experiment C)
====================================================================
Loads a 7B LLaMA checkpoint (full_attn or thin_keys) trained from scratch
and evaluates on standard benchmarks using lm-evaluation-harness.

Tasks: Hellaswag (10-shot), ARC-Challenge (25-shot), WinoGrande (5-shot),
       MMLU (5-shot), GSM8K (5-shot CoT)

Usage:
  # Full attention baseline
  python eval_downstream_7b_scratch.py --mode full_attn --device cuda:0

  # Thin keys
  python eval_downstream_7b_scratch.py --mode thin_keys --device cuda:4

  # Quick smoke test
  python eval_downstream_7b_scratch.py --mode full_attn --device cuda:0 \
      --tasks hellaswag --limit 100
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import lm_eval
from lm_eval.models.huggingface import HFLM

CKPT_DIR = "/sg-pretrain/checkpoints/expC_7b"
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
LOG_DIR = "/root/d_select/paper/experiments/logs"


# ════════════════════════════════════════════════════════════════
# RoPE helpers (same as experiment_c.py)
# ════════════════════════════════════════════════════════════════

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1) if cos.dim() == 3 else cos
    sin = sin.unsqueeze(1) if sin.dim() == 3 else sin
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def build_rope_cache(seq_len, head_dim, device, dtype, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    emb = torch.outer(t, freqs)
    cos = torch.cos(emb).to(dtype)
    sin = torch.sin(emb).to(dtype)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0)
    return cos, sin


# ════════════════════════════════════════════════════════════════
# Thin Keys patching (same as experiment_c.py)
# ════════════════════════════════════════════════════════════════

def patch_llama_thin_keys(model, d_select=1024):
    """Replace Q,K projections with smaller ones for thin keys."""
    cfg = model.config
    d_model = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    d_head_qk = d_select // n_heads
    d_head_v = d_model // n_heads

    n_patched = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.q_proj = nn.Linear(d_model, d_select, bias=False)
        attn.k_proj = nn.Linear(d_model, d_select, bias=False)

        attn._thin_cfg = dict(
            d_select=d_select, d_head_qk=d_head_qk,
            d_head_v=d_head_v, n_heads=n_heads, d_model=d_model,
        )

        def make_forward(a):
            def fwd(hidden_states, attention_mask=None, position_ids=None,
                    past_key_value=None, output_attentions=False,
                    use_cache=False, position_embeddings=None, **kw):
                c = a._thin_cfg
                B, T, _ = hidden_states.shape
                dev = hidden_states.device
                dt = hidden_states.dtype

                q = a.q_proj(hidden_states).view(B, T, c['n_heads'], c['d_head_qk']).transpose(1, 2)
                k = a.k_proj(hidden_states).view(B, T, c['n_heads'], c['d_head_qk']).transpose(1, 2)
                v = a.v_proj(hidden_states).view(B, T, c['n_heads'], c['d_head_v']).transpose(1, 2)

                cos, sin = build_rope_cache(T, c['d_head_qk'], dev, dt)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True,
                    scale=1.0 / math.sqrt(c['d_head_qk']),
                )
                out = out.transpose(1, 2).contiguous().view(B, T, c['d_model'])
                return (a.o_proj(out), None)
            return fwd

        attn.forward = make_forward(attn)
        n_patched += 1

    return n_patched


def load_model(mode, d_select=1024, device='cuda:0'):
    """Create model architecture and load checkpoint."""
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    config = LlamaConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        use_cache=True,  # enable KV cache for generation
    )
    config._attn_implementation = "sdpa"

    print(f"  Creating {mode} model architecture...")
    model = LlamaForCausalLM(config)

    if mode == "thin_keys":
        n = patch_llama_thin_keys(model, d_select=d_select)
        print(f"  Patched {n} layers with thin keys (d_select={d_select})")

    # Load checkpoint
    tag = f"thin{d_select}" if mode == "thin_keys" else "full_attn"
    ckpt_path = os.path.join(CKPT_DIR, f"expC_7b_{tag}.pt")
    print(f"  Loading checkpoint from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict
    print(f"  Checkpoint loaded successfully")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_params:,} ({n_params / 1e9:.2f}B)")

    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Downstream eval for 7B from-scratch models (Exp C)")
    parser.add_argument('--mode', choices=['full_attn', 'thin_keys'],
                        required=True)
    parser.add_argument('--d_select', type=int, default=1024)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--tasks', type=str,
                        default='mmlu,hellaswag,arc_challenge,winogrande,gsm8k',
                        help='Comma-separated list of tasks')
    parser.add_argument('--batch_size', type=str, default='auto')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit examples per task (for testing)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"Downstream Evaluation: 7B From-Scratch ({args.mode})")
    print("=" * 70)

    # Load model
    t0 = time.time()
    model, tokenizer = load_model(args.mode, d_select=args.d_select,
                                  device=args.device)
    print(f"  Model ready in {time.time()-t0:.1f}s\n")

    # Wrap in lm-eval HFLM
    print("Wrapping model for lm-eval...")
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Run evaluation
    task_list = [t.strip() for t in args.tasks.split(',')]
    print(f"\nEvaluating on: {task_list}")
    print(f"Limit: {args.limit if args.limit else 'None (full)'}")
    t0 = time.time()

    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    elapsed = time.time() - t0
    print(f"\nEvaluation completed in {elapsed/60:.1f} minutes")

    # Extract and print results
    tag = f"thin{args.d_select}" if args.mode == "thin_keys" else "full_attn"
    print(f"\n{'='*70}")
    print(f"RESULTS: 7B from-scratch ({args.mode})")
    print(f"{'='*70}")
    print(f"{'Task':<20} {'Metric':<30} {'Score':>10}")
    print("-" * 60)

    summary = {
        'experiment': 'C',
        'model': f'7b_scratch_{args.mode}',
        'mode': args.mode,
        'd_select': args.d_select if args.mode == 'thin_keys' else None,
        'tasks': {},
        'elapsed_minutes': round(elapsed / 60, 1),
    }

    for task_name, task_results in results['results'].items():
        task_summary = {}
        for metric, value in sorted(task_results.items()):
            if 'alias' in metric:
                continue
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
    os.makedirs(LOG_DIR, exist_ok=True)
    save_path = os.path.join(LOG_DIR, f'expC_downstream_{tag}.json')
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {save_path}")

    # Save full lm-eval results
    full_path = os.path.join(LOG_DIR, f'expC_downstream_{tag}_full.json')
    serializable = {k: v for k, v in results.items() if k in ['results', 'configs']}
    with open(full_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"Saved full results to {full_path}")


if __name__ == '__main__':
    main()

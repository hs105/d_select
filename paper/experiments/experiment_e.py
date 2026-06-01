#!/usr/bin/env python3
"""
Experiment E: Chinchilla-Optimal Scaling Laws for Thin Keys
============================================================

Train thin_keys vs full_attn at Chinchilla-optimal token budgets (~20 tokens/param)
across three model sizes (125M, 350M, 1.3B). Goal: characterize how the thin-keys
PPL gap scales with model size when the model is capacity-limited.

Model configs (LLaMA architecture):
  125M:  d=768,  h=12, L=12, d_ff=2048,  tokens=2.5B
  350M:  d=1024, h=16, L=24, d_ff=2816,  tokens=7B
  1.3B:  d=2048, h=16, L=24, d_ff=5504,  tokens=26B

All use d_select = d_model/4 for thin keys (consistent with Exp C/C2).

Usage:
    # Train 125M (smoke test, ~2 hours on 4 GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        experiment_e.py --scale 125M --mode full_attn

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
        experiment_e.py --scale 125M --mode thin_keys

    # Train 350M (~12 hours on 4 GPUs)
    ... --scale 350M --mode full_attn / thin_keys

    # Train 1.3B (~3 days on 4 GPUs)
    ... --scale 1.3B --mode full_attn / thin_keys

    # Resume from checkpoint
    ... --scale 1.3B --mode thin_keys --resume
"""

import argparse
import functools
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from experiment_c import (
    Logger,
    BinDataset,
    load_wikitext_validation,
    patch_llama_thin_keys,
    evaluate,
    TOKENIZER_NAME,
    LOG_DIR,
)

TOKEN_FILE = "/sg-pretrain/datasets/owt_tokens_full.bin"
CKPT_BASE = "/sg-pretrain/checkpoints/expE_scaling"

# ════════════════════════════════════════════════════════════════
# Model configurations
# ════════════════════════════════════════════════════════════════

SCALE_CONFIGS = {
    "125M": dict(
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=2048,
        d_select=192,           # d_model/4
        total_tokens=2_500_000_000,
        batch_size=8,           # larger batch OK for smaller model
        accumulation_steps=4,
        lr=6e-4,
        warmup_steps=1000,
        eval_interval=500,
        ckpt_interval=2000,
    ),
    "350M": dict(
        hidden_size=1024,
        num_attention_heads=16,
        num_hidden_layers=24,
        intermediate_size=2816,
        d_select=256,           # d_model/4
        total_tokens=7_000_000_000,
        batch_size=4,
        accumulation_steps=8,
        lr=3e-4,
        warmup_steps=2000,
        eval_interval=1000,
        ckpt_interval=5000,
    ),
    "1.3B": dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=24,
        intermediate_size=5504,
        d_select=512,           # d_model/4
        total_tokens=26_000_000_000,
        batch_size=8,
        accumulation_steps=8,
        lr=2e-4,
        warmup_steps=2000,
        eval_interval=2000,
        ckpt_interval=10000,
    ),
}


# ════════════════════════════════════════════════════════════════
# Model creation (generalized from experiment_c.create_model)
# ════════════════════════════════════════════════════════════════

def create_model(scale, mode, seed=42):
    """Create LLaMA model at the given scale, optionally with thin keys."""
    from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer
    import torch.nn as nn

    cfg = SCALE_CONFIGS[scale]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    d_model = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    d_select = cfg["d_select"]

    model_config = LlamaConfig(
        hidden_size=d_model,
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,  # MHA (no GQA)
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        tie_word_embeddings=True,     # tie for smaller models
        use_cache=False,
    )
    model_config._attn_implementation = "sdpa"

    torch.manual_seed(seed)
    model = LlamaForCausalLM(model_config)

    n_thin_params_saved = 0
    if mode == "thin_keys":
        orig_qk = sum(
            layer.self_attn.q_proj.weight.numel() +
            layer.self_attn.k_proj.weight.numel()
            for layer in model.model.layers
        )
        n = patch_llama_thin_keys(model, d_select=d_select)
        new_qk = sum(
            layer.self_attn.q_proj.weight.numel() +
            layer.self_attn.k_proj.weight.numel()
            for layer in model.model.layers
        )
        n_thin_params_saved = orig_qk - new_qk
        print(f"  Patched {n} layers with thin keys "
              f"(d_select={d_select}, d_head_qk={d_select // n_heads})")
        print(f"  Q,K params: {orig_qk:,} -> {new_qk:,} "
              f"(saved {n_thin_params_saved:,})")
    else:
        print(f"  Full attention (SDPA)")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_total:,} ({n_total / 1e6:.1f}M)")

    return model, tokenizer, n_total, n_thin_params_saved


# ════════════════════════════════════════════════════════════════
# Checkpoint management (adapted from experiment_c2)
# ════════════════════════════════════════════════════════════════

def find_latest_checkpoint(prefix, ckpt_dir):
    """Find the latest checkpoint matching prefix_step{N}.pt."""
    pattern = os.path.join(ckpt_dir, f"{prefix}_step*.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None, None

    def extract_step(path):
        return int(os.path.basename(path).rsplit("_step", 1)[-1].replace(".pt", ""))

    matches.sort(key=extract_step)
    latest = matches[-1]
    meta_path = latest.replace(".pt", ".json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {"step": extract_step(latest), "epoch": 0,
                "best_ppl": float("inf"), "best_step": 0, "trajectory": []}
    return latest, meta


def cleanup_old_checkpoints(prefix, ckpt_dir, keep=2):
    """Keep only the last `keep` checkpoints."""
    pattern = os.path.join(ckpt_dir, f"{prefix}_step*.pt")
    matches = sorted(glob.glob(pattern))
    if len(matches) <= keep:
        return
    for old in matches[:-keep]:
        os.remove(old)
        meta = old.replace(".pt", ".json")
        if os.path.exists(meta):
            os.remove(meta)


def save_checkpoint(model, prefix, step, epoch, best_ppl, best_step,
                    trajectory, rank, logger, ckpt_dir, final=False,
                    use_fsdp=True):
    """Save model checkpoint."""
    label = "final" if final else "periodic"
    logger.log(f"\n  Saving {label} checkpoint at step {step}...")

    if use_fsdp:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    if rank == 0:
        ckpt_path = os.path.join(ckpt_dir, f"{prefix}_step{step}.pt")
        torch.save(state_dict, ckpt_path)
        size_gb = os.path.getsize(ckpt_path) / 1e9
        logger.log(f"  Saved {ckpt_path} ({size_gb:.1f} GB)")

        meta = {"step": step, "epoch": epoch, "best_ppl": best_ppl,
                "best_step": best_step, "trajectory": trajectory}
        with open(ckpt_path.replace(".pt", ".json"), "w") as f:
            json.dump(meta, f, indent=2)

        if not final:
            cleanup_old_checkpoints(prefix, ckpt_dir, keep=2)

    if dist.is_initialized():
        dist.barrier()
    del state_dict


# ════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════

def train(args):
    cfg = SCALE_CONFIGS[args.scale]

    # ── Distributed setup ──
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tag = f"thin{cfg['d_select']}" if args.mode == "thin_keys" else "full_attn"
    prefix = f"expE_{args.scale}_{tag}"
    seed_suffix = f"_s{args.seed}" if args.seed != 42 else ""
    ckpt_dir = os.path.join(args.ckpt_base, f"{args.scale}_{tag}")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{prefix}{seed_suffix}.log")
    logger = Logger(log_path, rank=rank)

    # ── Compute schedule ──
    batch_size = cfg["batch_size"]
    accum = cfg["accumulation_steps"]
    total_tokens = cfg["total_tokens"]
    seq_len = args.seq_len
    eff_batch = world_size * batch_size * accum
    tok_per_step = eff_batch * seq_len
    total_steps = total_tokens // tok_per_step

    logger.log("=" * 70)
    logger.log(f"EXPERIMENT E — SCALING LAWS: {args.scale} {args.mode.upper()}")
    logger.log("=" * 70)
    logger.log(f"  Scale: {args.scale}  |  d_model: {cfg['hidden_size']}  |  "
               f"heads: {cfg['num_attention_heads']}  |  layers: {cfg['num_hidden_layers']}")
    logger.log(f"  GPUs: {world_size}")
    logger.log(f"  Batch: {batch_size}/gpu x {accum} accum "
               f"x {world_size} gpus = {eff_batch}")
    logger.log(f"  Tokens/step: {tok_per_step:,}  |  Total steps: {total_steps:,}")
    logger.log(f"  Tokens: {total_tokens / 1e9:.1f}B  |  Seq len: {seq_len}")
    logger.log(f"  LR: {cfg['lr']}  |  Warmup: {cfg['warmup_steps']}")
    logger.log(f"  Seed: {args.seed}")
    tok_per_param = total_tokens / (cfg['hidden_size'] ** 2 * cfg['num_hidden_layers'] * 12)
    logger.log(f"  Tokens/param ratio: ~{total_tokens / 1e9:.1f}B / ~{args.scale}")
    if args.mode == "thin_keys":
        logger.log(f"  d_select: {cfg['d_select']} (d_model/4)")

    # ── Data ──
    token_file = args.token_file
    if not os.path.exists(token_file):
        raise FileNotFoundError(
            f"Token file not found: {token_file}\n"
            f"Run: python experiment_c2.py --prepare_data")

    n_file_tokens = os.path.getsize(token_file) // 2
    n_epochs_needed = math.ceil(total_tokens / n_file_tokens)
    logger.log(f"  Token file: {token_file} ({n_file_tokens / 1e9:.2f}B tokens)")
    logger.log(f"  Epochs needed: ~{n_epochs_needed}")

    split = int(n_file_tokens * 0.95)
    train_ds = BinDataset(token_file, seq_len, 0, split)
    val_ds = BinDataset(token_file, seq_len, split, n_file_tokens)
    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True) \
        if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False) \
        if dist.is_initialized() else None
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            sampler=val_sampler, num_workers=2, pin_memory=True)

    logger.log(f"  OWT Train: {len(train_ds):,} seqs  |  OWT Val: {len(val_ds):,} seqs")

    # ── WikiText-103 validation ──
    wt_val_loader = None
    if rank == 0:
        logger.log("  Loading WikiText-103 validation...")
    try:
        from transformers import AutoTokenizer
        wt_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        wt_val_ds = load_wikitext_validation(wt_tokenizer, seq_len=seq_len)
        wt_val_sampler = DistributedSampler(wt_val_ds, world_size, rank, shuffle=False) \
            if dist.is_initialized() else None
        wt_val_loader = DataLoader(wt_val_ds, batch_size=batch_size,
                                   sampler=wt_val_sampler, num_workers=2,
                                   pin_memory=True)
        logger.log(f"  WikiText-103 Val: {len(wt_val_ds):,} seqs")
    except Exception as e:
        logger.log(f"  WARNING: Could not load WikiText-103: {e}")

    # ── Model ──
    logger.log(f"\n  Creating {args.scale} {args.mode} model...")
    model, tokenizer, n_params, n_saved = create_model(
        args.scale, args.mode, seed=args.seed)

    # ── Resume ──
    start_step = 0
    start_epoch = 0
    best_ppl = float("inf")
    best_step = 0
    trajectory = []

    if args.resume:
        ckpt_path, meta = find_latest_checkpoint(prefix, ckpt_dir)
        if ckpt_path is not None:
            logger.log(f"\n  Resuming from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            del state_dict
            start_step = meta["step"]
            start_epoch = meta.get("epoch", 0)
            best_ppl = meta.get("best_ppl", float("inf"))
            best_step = meta.get("best_step", 0)
            trajectory = meta.get("trajectory", [])
            logger.log(f"  Resumed: step={start_step}, best_ppl={best_ppl:.2f}")
        else:
            logger.log(f"  --resume but no checkpoint found. Starting fresh.")

    # ── FSDP wrapping ──
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer},
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(model, auto_wrap_policy=wrap_policy, mixed_precision=mp,
                 sharding_strategy=ShardingStrategy.FULL_SHARD,
                 device_id=local_rank)

    logger.log(f"  FSDP ready | {n_params / 1e6:.1f}M params | "
               f"thin saved: {n_saved:,}")
    logger.log(f"  GPU mem: {torch.cuda.memory_allocated(device) / 1e9:.1f} GB")

    # ── Optimizer + scheduler ──
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=0.1, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < cfg["warmup_steps"]:
            return step / max(1, cfg["warmup_steps"])
        prog = (step - cfg["warmup_steps"]) / max(1, total_steps - cfg["warmup_steps"])
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    if start_step > 0:
        logger.log(f"  Fast-forwarding LR to step {start_step}...")
        for _ in range(start_step):
            sched.step()
        logger.log(f"  LR at resume: {sched.get_last_lr()[0]:.2e}")

    # ── Training loop ──
    eval_interval = cfg["eval_interval"]
    ckpt_interval = cfg["ckpt_interval"]
    step = start_step
    micro = 0
    t0 = time.time()
    pbar = tqdm(total=total_steps, initial=start_step,
                desc=f"{args.scale} {args.mode}", disable=(rank != 0))

    model.train()
    epoch = start_epoch
    opt.zero_grad()

    micros_per_step = accum
    skip_micros = 0
    if start_step > 0:
        skip_micros = start_step * micros_per_step
        if skip_micros >= len(train_loader):
            epoch = skip_micros // len(train_loader)
            skip_micros = skip_micros % len(train_loader)

    while step < total_steps:
        if train_sampler:
            train_sampler.set_epoch(epoch)

        micro_in_epoch = 0
        for x, y in train_loader:
            if step >= total_steps:
                break

            if epoch == start_epoch and micro_in_epoch < skip_micros:
                micro_in_epoch += 1
                continue
            micro_in_epoch += 1

            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out[0]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / accum).backward()

            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1

                if rank == 0:
                    pbar.set_postfix(
                        loss=f"{loss.item():.3f}",
                        lr=f"{sched.get_last_lr()[0]:.1e}",
                        ppl=f"{best_ppl:.1f}",
                        ep=epoch)
                    pbar.update(1)

                # ── Eval ──
                if step % eval_interval == 0 or step >= total_steps:
                    vl, vp = evaluate(model, val_loader, device,
                                      max_batches=50)
                    if dist.is_initialized():
                        t = torch.tensor([vl], device=device)
                        dist.all_reduce(t, op=dist.ReduceOp.AVG)
                        vl = t.item()
                        vp = math.exp(min(vl, 20))

                    wt_vl, wt_vp = None, None
                    if wt_val_loader is not None:
                        wt_vl, wt_vp = evaluate(model, wt_val_loader, device,
                                                max_batches=50)
                        if dist.is_initialized():
                            t = torch.tensor([wt_vl], device=device)
                            dist.all_reduce(t, op=dist.ReduceOp.AVG)
                            wt_vl = t.item()
                            wt_vp = math.exp(min(wt_vl, 20))

                    hrs = (time.time() - t0) / 3600
                    toks = step * tok_per_step

                    log_msg = (f"  step={step:>6d}  owt_ppl={vp:.2f}"
                               f"  owt_loss={vl:.4f}")
                    if wt_vp is not None:
                        log_msg += f"  wt103_ppl={wt_vp:.2f}"
                    log_msg += (f"  lr={sched.get_last_lr()[0]:.1e}"
                                f"  tok={toks / 1e9:.2f}B  ep={epoch}"
                                f"  t={hrs:.1f}h")
                    logger.log(log_msg)

                    entry = dict(step=step, owt_val_ppl=vp, owt_val_loss=vl,
                                 tokens=toks, hours=hrs, epoch=epoch)
                    if wt_vp is not None:
                        entry["wt103_val_ppl"] = wt_vp
                        entry["wt103_val_loss"] = wt_vl
                    trajectory.append(entry)

                    if vp < best_ppl:
                        best_ppl, best_step = vp, step
                        logger.log(f"  * New best OWT PPL={vp:.2f} at step {step}")

                    if rank == 0:
                        traj_path = os.path.join(
                            LOG_DIR, f"{prefix}{seed_suffix}_trajectory.json")
                        with open(traj_path, "w") as f:
                            json.dump(trajectory, f, indent=2)

                    model.train()

                # ── Checkpoint ──
                if step % ckpt_interval == 0 and step > start_step:
                    save_checkpoint(model, prefix, step, epoch,
                                    best_ppl, best_step, trajectory,
                                    rank, logger, ckpt_dir)

        epoch += 1
        skip_micros = 0

    pbar.close()

    # ── Final checkpoint ──
    save_checkpoint(model, prefix, step, epoch,
                    best_ppl, best_step, trajectory,
                    rank, logger, ckpt_dir, final=True)

    # ── Summary ──
    total_hrs = (time.time() - t0) / 3600
    logger.log(f"\n{'=' * 70}")
    logger.log(f"DONE: {args.scale} {args.mode.upper()}")
    logger.log(f"  Best OWT PPL: {best_ppl:.2f} at step {best_step}")
    if trajectory and "wt103_val_ppl" in trajectory[-1]:
        logger.log(f"  Final WT-103 PPL: {trajectory[-1]['wt103_val_ppl']:.2f}")
    logger.log(f"  Time: {total_hrs:.1f} hours")
    logger.log(f"  Tokens: {total_tokens / 1e9:.1f}B  |  Epochs: {epoch}")
    logger.log(f"{'=' * 70}")

    if rank == 0:
        best_wt103 = None
        if any("wt103_val_ppl" in e for e in trajectory):
            best_wt103 = min(e["wt103_val_ppl"] for e in trajectory
                             if "wt103_val_ppl" in e)

        results = {
            "experiment": "E",
            "scale": args.scale,
            "mode": args.mode,
            "seed": args.seed,
            "best_owt_ppl": best_ppl,
            "best_owt_step": best_step,
            "best_wt103_ppl": best_wt103,
            "final_owt_ppl": trajectory[-1]["owt_val_ppl"] if trajectory else None,
            "final_wt103_ppl": trajectory[-1].get("wt103_val_ppl"),
            "total_hours": total_hrs,
            "total_tokens": total_tokens,
            "total_epochs": epoch,
            "n_params": n_params,
            "n_thin_params_saved": n_saved,
            "model_config": {
                "hidden_size": cfg["hidden_size"],
                "num_heads": cfg["num_attention_heads"],
                "num_layers": cfg["num_hidden_layers"],
                "intermediate_size": cfg["intermediate_size"],
                "d_select": cfg["d_select"] if args.mode == "thin_keys" else cfg["hidden_size"],
            },
            "training_config": {
                "lr": cfg["lr"],
                "seq_len": seq_len,
                "batch_size": batch_size,
                "accum": accum,
                "world_size": world_size,
                "eff_batch": eff_batch,
            },
        }
        with open(os.path.join(LOG_DIR, f"{prefix}{seed_suffix}.json"), "w") as f:
            json.dump(results, f, indent=2)

    logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Experiment E: Chinchilla-optimal scaling laws for thin keys")
    p.add_argument("--scale", choices=["125M", "350M", "1.3B"], required=True,
                   help="Model scale")
    p.add_argument("--mode", choices=["full_attn", "thin_keys"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--token_file", type=str, default=TOKEN_FILE)
    p.add_argument("--ckpt_base", type=str, default=CKPT_BASE)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment C2: Train 7B LLaMA from Scratch — 20B Tokens
========================================================

Extends Experiment C to Chinchilla-optimal token regime.
Exp C showed thin keys beats full attention at 7B/2B tokens (ratio ~0.3,
overparameterized). This experiment trains on 20B tokens (ratio ~3.0)
to test whether the result holds beyond the overparameterized regime.

Training: ~305K steps × ~3 sec/step ≈ 10.6 days per config.
Both configs run in parallel on 4 GPUs each (8×H100 total).

Usage:
    # Step 1: Prepare full OWT data (~8B tokens, CPU-only, ~2-4 hours)
    python experiment_c2.py --prepare_data

    # Step 2: Train models in parallel on separate GPU sets
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        experiment_c2.py --mode full_attn

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
        experiment_c2.py --mode thin_keys

    # Resume from checkpoint after interruption
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        experiment_c2.py --mode full_attn --resume
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

# Import shared components from experiment_c
from experiment_c import (
    Logger,
    BinDataset,
    load_wikitext_validation,
    create_model,
    evaluate,
    TOKENIZER_NAME,
    LOG_DIR,
)

TOKEN_FILE = "/sg-pretrain/datasets/owt_tokens_full.bin"
CKPT_DIR = "/sg-pretrain/checkpoints/expC2_7b"  # default; overridden by --ckpt_dir


# ════════════════════════════════════════════════════════════════
# Data preparation (full OWT, ~8B tokens)
# ════════════════════════════════════════════════════════════════

def prepare_data():
    """Tokenize full OpenWebText via streaming (~8B tokens).

    Can run on CPU while GPUs are busy with other experiments.
    Writes incrementally in 50M-token chunks to avoid memory issues.
    Saves to /sg-pretrain/datasets/owt_tokens_full.bin (~16 GB).
    """
    if os.path.exists(TOKEN_FILE):
        n = os.path.getsize(TOKEN_FILE) // 2
        print(f"Token file already exists: {TOKEN_FILE} ({n:,} tokens, "
              f"{os.path.getsize(TOKEN_FILE) / 1e9:.1f} GB)")
        # Full OWT is ~8B tokens; accept anything >= 7.5B
        if n >= 7_500_000_000:
            print("  Sufficient for 20B training (will do ~2.5 epochs). Skipping.")
            return
        print(f"  Only {n:,} tokens — need full OWT. Re-creating.")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Preparing full OpenWebText (all tokens)...")
    print(f"  Output: {TOKEN_FILE}")
    print(f"  Writing incrementally in 50M-token chunks")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    tmp_file = TOKEN_FILE + ".tmp"
    ds = load_dataset("openwebtext", split="train", streaming=True)

    CHUNK_SIZE = 50_000_000  # flush every 50M tokens
    chunk_ids = []
    total_tokens = 0
    t0 = time.time()

    with open(tmp_file, "wb") as f:
        for i, sample in enumerate(tqdm(ds, desc="Tokenizing OWT")):
            ids = tokenizer.encode(sample["text"])
            chunk_ids.extend(ids)

            if len(chunk_ids) >= CHUNK_SIZE:
                arr = np.array(chunk_ids, dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(chunk_ids)
                chunk_ids = []

                elapsed = time.time() - t0
                rate = total_tokens / elapsed
                print(f"  {total_tokens / 1e9:.2f}B tokens, "
                      f"{rate / 1e6:.1f}M tok/s, "
                      f"elapsed {elapsed / 3600:.1f}h, "
                      f"file {os.path.getsize(tmp_file) / 1e9:.1f} GB")

        # Write remaining tokens
        if chunk_ids:
            arr = np.array(chunk_ids, dtype=np.uint16)
            arr.tofile(f)
            total_tokens += len(chunk_ids)

    # Atomic rename
    os.rename(tmp_file, TOKEN_FILE)
    elapsed = time.time() - t0
    print(f"Saved {total_tokens:,} tokens to {TOKEN_FILE} "
          f"({os.path.getsize(TOKEN_FILE) / 1e9:.1f} GB, {elapsed / 60:.0f} min)")


# ════════════════════════════════════════════════════════════════
# Checkpoint management
# ════════════════════════════════════════════════════════════════

def find_latest_checkpoint(tag, ckpt_dir=CKPT_DIR):
    """Find the latest periodic checkpoint for a given tag.

    Returns (ckpt_path, meta_dict) or (None, None) if no checkpoint found.
    Checkpoint naming: expC2_7b_{tag}_step{step}.pt
    Metadata sidecar:  expC2_7b_{tag}_step{step}.json
    """
    pattern = os.path.join(ckpt_dir, f"expC2_7b_{tag}_step*.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None, None

    # Extract step numbers and find the latest
    def extract_step(path):
        base = os.path.basename(path)
        # expC2_7b_{tag}_step{N}.pt
        step_str = base.rsplit("_step", 1)[-1].replace(".pt", "")
        return int(step_str)

    matches.sort(key=extract_step)
    latest = matches[-1]
    step = extract_step(latest)

    # Load metadata sidecar
    meta_path = latest.replace(".pt", ".json")
    meta = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        # Minimal metadata if sidecar missing
        meta = {"step": step, "epoch": 0, "best_ppl": float("inf"),
                "best_step": 0, "trajectory": []}

    return latest, meta


def cleanup_old_checkpoints(tag, keep=2, ckpt_dir=CKPT_DIR):
    """Keep only the last `keep` checkpoints, delete older ones."""
    pattern = os.path.join(ckpt_dir, f"expC2_7b_{tag}_step*.pt")
    matches = sorted(glob.glob(pattern))
    if len(matches) <= keep:
        return

    for old in matches[:-keep]:
        os.remove(old)
        # Also remove sidecar JSON
        meta = old.replace(".pt", ".json")
        if os.path.exists(meta):
            os.remove(meta)


# ════════════════════════════════════════════════════════════════
# Training
# ════════════════════════════════════════════════════════════════

def train(args):
    # ── Distributed setup ──
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tag = f"thin{args.d_select}" if args.mode == "thin_keys" else "full_attn"
    ckpt_dir = os.path.join(args.ckpt_dir, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    seed_suffix = f"_s{args.seed}" if args.seed != 42 else ""
    log_path = os.path.join(LOG_DIR, f"expC2_7b_{tag}{seed_suffix}.log")
    logger = Logger(log_path, rank=rank)

    # ── Compute schedule ──
    eff_batch = world_size * args.batch_size * args.accumulation_steps
    tok_per_step = eff_batch * args.seq_len
    total_steps = args.total_tokens // tok_per_step

    logger.log("=" * 70)
    logger.log(f"EXPERIMENT C2 — FROM-SCRATCH 7B, 20B TOKENS: {args.mode.upper()}")
    logger.log("=" * 70)
    logger.log(f"  GPUs: {world_size}")
    logger.log(f"  Batch: {args.batch_size}/gpu × {args.accumulation_steps} accum "
               f"× {world_size} gpus = {eff_batch}")
    logger.log(f"  Tokens/step: {tok_per_step:,}  |  Total steps: {total_steps:,}")
    logger.log(f"  Tokens: {args.total_tokens / 1e9:.1f}B  |  Seq len: {args.seq_len}")
    logger.log(f"  LR: {args.lr} → {args.lr / 10}  |  Warmup: {args.warmup_steps}")
    logger.log(f"  Eval interval: {args.eval_interval}  |  "
               f"Ckpt interval: {args.ckpt_interval}")
    if args.mode == "thin_keys":
        logger.log(f"  d_select: {args.d_select} (d_model/4)")

    # ── Resolve token file ──
    token_file = args.token_file
    if not os.path.exists(token_file):
        fallback = "/sg-pretrain/datasets/owt_tokens_2B.bin"
        if os.path.exists(fallback):
            logger.log(f"  WARNING: {token_file} not found, falling back to {fallback}")
            token_file = fallback
        else:
            raise FileNotFoundError(
                f"Token file not found: {token_file}\n"
                f"Run: python experiment_c2.py --prepare_data")

    n_file_tokens = os.path.getsize(token_file) // 2
    n_epochs_needed = math.ceil(args.total_tokens / n_file_tokens)
    logger.log(f"  Token file: {token_file} ({n_file_tokens / 1e9:.2f}B tokens)")
    logger.log(f"  Epochs needed: ~{n_epochs_needed} "
               f"(ratio: {args.total_tokens / n_file_tokens:.1f}x)")

    # ── Data ──
    split = int(n_file_tokens * 0.95)
    train_ds = BinDataset(token_file, args.seq_len, 0, split)
    val_ds = BinDataset(token_file, args.seq_len, split, n_file_tokens)
    train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True) \
        if dist.is_initialized() else None
    val_sampler = DistributedSampler(val_ds, world_size, rank, shuffle=False) \
        if dist.is_initialized() else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            sampler=val_sampler, num_workers=2, pin_memory=True)

    logger.log(f"  OWT Train: {len(train_ds):,} seqs  |  OWT Val: {len(val_ds):,} seqs")

    # ── WikiText-103 validation ──
    wt_val_loader = None
    if rank == 0:
        logger.log("  Loading WikiText-103 validation...")
    try:
        from transformers import AutoTokenizer
        wt_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        wt_val_ds = load_wikitext_validation(wt_tokenizer, seq_len=args.seq_len)
        wt_val_sampler = DistributedSampler(wt_val_ds, world_size, rank, shuffle=False) \
            if dist.is_initialized() else None
        wt_val_loader = DataLoader(wt_val_ds, batch_size=args.batch_size,
                                   sampler=wt_val_sampler, num_workers=2,
                                   pin_memory=True)
        logger.log(f"  WikiText-103 Val: {len(wt_val_ds):,} seqs")
    except Exception as e:
        logger.log(f"  WARNING: Could not load WikiText-103: {e}")
        logger.log(f"  Will use OWT validation only.")

    # ── Model (created on CPU) ──
    logger.log(f"\n  Creating {args.mode} model...")
    model, tokenizer, n_params, n_saved = create_model(
        args.mode, d_select=args.d_select, seed=args.seed)

    # ── Resume: load state dict BEFORE FSDP wrapping ──
    start_step = 0
    start_epoch = 0
    best_ppl = float("inf")
    best_step = 0
    trajectory = []

    if args.resume:
        ckpt_path, meta = find_latest_checkpoint(tag, ckpt_dir=ckpt_dir)
        if ckpt_path is not None:
            logger.log(f"\n  Resuming from checkpoint: {ckpt_path}")
            logger.log(f"  Loading state dict on CPU...")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            del state_dict  # free memory

            start_step = meta["step"]
            start_epoch = meta.get("epoch", 0)
            best_ppl = meta.get("best_ppl", float("inf"))
            best_step = meta.get("best_step", 0)
            trajectory = meta.get("trajectory", [])
            logger.log(f"  Resumed: step={start_step}, epoch={start_epoch}, "
                       f"best_ppl={best_ppl:.2f}")
        else:
            logger.log(f"  --resume specified but no checkpoint found. "
                       f"Starting from scratch.")

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

    logger.log(f"  FSDP ready | {n_params / 1e9:.2f}B params | "
               f"thin params saved: {n_saved:,}")
    logger.log(f"  GPU memory after FSDP: "
               f"{torch.cuda.memory_allocated(device) / 1e9:.1f} GB")

    # ── Optimizer + scheduler ──
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.1, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Fast-forward scheduler if resuming
    if start_step > 0:
        logger.log(f"  Fast-forwarding LR scheduler to step {start_step}...")
        for _ in range(start_step):
            sched.step()
        logger.log(f"  LR at resume point: {sched.get_last_lr()[0]:.2e}")

    # ── Training loop ──
    os.makedirs(ckpt_dir, exist_ok=True)
    step = start_step
    micro = 0
    t0 = time.time()
    pbar = tqdm(total=total_steps, initial=start_step,
                desc=args.mode, disable=(rank != 0))

    model.train()
    epoch = start_epoch
    opt.zero_grad()

    # Compute how many micro-batches to skip in the first epoch on resume
    # (within the epoch we were interrupted in)
    micros_per_step = args.accumulation_steps
    steps_per_epoch = len(train_loader) // micros_per_step
    skip_micros_in_first_epoch = 0
    if start_step > 0 and start_epoch > 0:
        # We completed start_epoch full epochs; no skip needed in current epoch
        pass
    elif start_step > 0:
        # Approximate: skip microbatches already processed in epoch 0
        skip_micros_in_first_epoch = start_step * micros_per_step
        if skip_micros_in_first_epoch >= len(train_loader):
            # Already past this epoch, advance
            epoch = skip_micros_in_first_epoch // len(train_loader)
            skip_micros_in_first_epoch = skip_micros_in_first_epoch % len(train_loader)

    while step < total_steps:
        if train_sampler:
            train_sampler.set_epoch(epoch)

        micro_in_epoch = 0
        for x, y in train_loader:
            if step >= total_steps:
                break

            # Skip already-processed microbatches on resume
            if epoch == start_epoch and micro_in_epoch < skip_micros_in_first_epoch:
                micro_in_epoch += 1
                continue
            micro_in_epoch += 1

            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out.logits if hasattr(out, "logits") else out[0]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / args.accumulation_steps).backward()

            micro += 1
            if micro % args.accumulation_steps == 0:
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
                if step % args.eval_interval == 0 or step >= total_steps:
                    vl, vp = evaluate(model, val_loader, device,
                                      max_batches=args.eval_batches)

                    if dist.is_initialized():
                        t = torch.tensor([vl], device=device)
                        dist.all_reduce(t, op=dist.ReduceOp.AVG)
                        vl = t.item()
                        vp = math.exp(min(vl, 20))

                    wt_vl, wt_vp = None, None
                    if wt_val_loader is not None:
                        wt_vl, wt_vp = evaluate(model, wt_val_loader, device,
                                                max_batches=args.eval_batches)
                        if dist.is_initialized():
                            t = torch.tensor([wt_vl], device=device)
                            dist.all_reduce(t, op=dist.ReduceOp.AVG)
                            wt_vl = t.item()
                            wt_vp = math.exp(min(wt_vl, 20))

                    hrs = (time.time() - t0) / 3600
                    toks = step * tok_per_step

                    log_msg = (
                        f"  step={step:>6d}  owt_ppl={vp:.2f}  owt_loss={vl:.4f}")
                    if wt_vp is not None:
                        log_msg += f"  wt103_ppl={wt_vp:.2f}"
                    log_msg += (
                        f"  lr={sched.get_last_lr()[0]:.1e}  "
                        f"tok={toks / 1e9:.2f}B  ep={epoch}  t={hrs:.1f}h")
                    logger.log(log_msg)

                    entry = dict(
                        step=step, owt_val_ppl=vp, owt_val_loss=vl,
                        tokens=toks, hours=hrs, epoch=epoch)
                    if wt_vp is not None:
                        entry["wt103_val_ppl"] = wt_vp
                        entry["wt103_val_loss"] = wt_vl
                    trajectory.append(entry)

                    if vp < best_ppl:
                        best_ppl, best_step = vp, step
                        logger.log(f"  ★ New best OWT PPL={vp:.2f} at step {step}")

                    if rank == 0:
                        with open(os.path.join(
                                LOG_DIR,
                                f"expC2_7b_{tag}_trajectory.json"), "w") as f:
                            json.dump(trajectory, f, indent=2)

                    model.train()

                # ── Periodic checkpoint ──
                if step % args.ckpt_interval == 0 and step > start_step:
                    _save_checkpoint(model, tag, step, epoch,
                                     best_ppl, best_step, trajectory,
                                     rank, logger, ckpt_dir=ckpt_dir)

        epoch += 1
        # After first resumed epoch, no more skipping
        skip_micros_in_first_epoch = 0

    pbar.close()

    # ── Final checkpoint ──
    _save_checkpoint(model, tag, step, epoch,
                     best_ppl, best_step, trajectory,
                     rank, logger, final=True, ckpt_dir=ckpt_dir)

    # ── Summary ──
    total_hrs = (time.time() - t0) / 3600
    logger.log(f"\n{'=' * 70}")
    logger.log(f"DONE: {args.mode.upper()}")
    logger.log(f"  Best OWT PPL: {best_ppl:.2f} at step {best_step}")
    if trajectory and "wt103_val_ppl" in trajectory[-1]:
        logger.log(f"  Final WT-103 PPL: {trajectory[-1]['wt103_val_ppl']:.2f}")
    logger.log(f"  Time: {total_hrs:.1f} hours")
    logger.log(f"  Tokens: {args.total_tokens / 1e9:.1f}B  |  Epochs: {epoch}")
    logger.log(f"{'=' * 70}")

    if rank == 0:
        best_wt103 = None
        if any("wt103_val_ppl" in e for e in trajectory):
            best_wt103 = min(e["wt103_val_ppl"] for e in trajectory
                             if "wt103_val_ppl" in e)

        results = {
            "experiment": "C2",
            "mode": args.mode,
            "best_owt_ppl": best_ppl,
            "best_owt_step": best_step,
            "best_wt103_ppl": best_wt103,
            "final_owt_ppl": trajectory[-1]["owt_val_ppl"] if trajectory else None,
            "final_wt103_ppl": trajectory[-1].get("wt103_val_ppl"),
            "total_hours": total_hrs,
            "total_tokens": args.total_tokens,
            "total_epochs": epoch,
            "n_params": n_params,
            "n_thin_params_saved": n_saved,
            "config": dict(
                lr=args.lr, seq_len=args.seq_len,
                batch_size=args.batch_size,
                accum=args.accumulation_steps,
                world_size=world_size,
                eff_batch=eff_batch,
            ),
        }
        if args.mode == "thin_keys":
            results["thin_keys"] = dict(
                d_select=args.d_select,
                d_head_qk=args.d_select // 32,
                d_head_v=128,
                n_heads=32,
            )

        with open(os.path.join(LOG_DIR, f"expC2_7b_{tag}.json"), "w") as f:
            json.dump(results, f, indent=2)

    logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()


def _save_checkpoint(model, tag, step, epoch, best_ppl, best_step,
                     trajectory, rank, logger, final=False, ckpt_dir=CKPT_DIR):
    """Save model checkpoint using FSDP full state dict gather."""
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        FullStateDictConfig,
        StateDictType,
    )

    label = "final" if final else "periodic"
    logger.log(f"\n  Saving {label} checkpoint at step {step}...")

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.state_dict()

    if rank == 0:
        ckpt_path = os.path.join(ckpt_dir, f"expC2_7b_{tag}_step{step}.pt")
        torch.save(state_dict, ckpt_path)
        size_gb = os.path.getsize(ckpt_path) / 1e9
        logger.log(f"  Saved {ckpt_path} ({size_gb:.1f} GB)")

        # Save metadata sidecar
        meta = {
            "step": step,
            "epoch": epoch,
            "best_ppl": best_ppl,
            "best_step": best_step,
            "trajectory": trajectory,
        }
        meta_path = ckpt_path.replace(".pt", ".json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Keep only last 2 periodic checkpoints; never delete the final one
        if not final:
            cleanup_old_checkpoints(tag, keep=2, ckpt_dir=ckpt_dir)

    if dist.is_initialized():
        dist.barrier()

    del state_dict


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Experiment C2: 7B from scratch — 20B tokens")
    p.add_argument("--prepare_data", action="store_true",
                   help="Tokenize full OWT (~8B tokens). CPU-only, no GPU needed.")
    p.add_argument("--mode", choices=["full_attn", "thin_keys"],
                   default="full_attn")
    p.add_argument("--d_select", type=int, default=1024,
                   help="QK projection dim for thin_keys (d_model/4 = 1024)")
    p.add_argument("--total_tokens", type=int, default=20_000_000_000,
                   help="Total training tokens (default: 20B)")
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Per-GPU micro-batch size")
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=5000,
                   help="Steps between eval (default: 5000 → ~61 eval points)")
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=10000,
                   help="Steps between periodic checkpoints (default: 10000 → ~8h)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--ckpt_dir", type=str, default=CKPT_DIR,
                   help="Checkpoint directory (default: /sg-pretrain/checkpoints/expC2_7b)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint")
    p.add_argument("--token_file", type=str, default=TOKEN_FILE,
                   help="Path to token file (default: full OWT)")
    args = p.parse_args()

    if args.prepare_data:
        prepare_data()
        return

    train(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Experiment C: Train 7B LLaMA from Scratch — Full Attention vs Thin Keys
=======================================================================

Trains a LLaMA-7B model from random initialization on OpenWebText (2B tokens).
Compares two attention modes:
  - full_attn:  Standard full causal attention (SDPA / Flash)
  - thin_keys:  Thin QK projections (d_select=1024 = d_model/4)

Addresses reviewer W3: train-from-scratch validation at 7B scale.
Prior experiments show ~4% PPL cost at d_select=d_model/4 for 10M/125M.

Uses FSDP for memory-efficient training (fp32 optimizer, bf16 forward).

Usage:
    # Step 1: Prepare data (run once, ~30-60 min)
    python experiment_c.py --prepare_data

    # Step 2: Train models in parallel on separate GPU sets
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        experiment_c.py --mode full_attn

    CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
        experiment_c.py --mode thin_keys
"""

import argparse
import functools
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

TOKEN_FILE = "/sg-pretrain/datasets/owt_tokens_2B.bin"
CKPT_DIR = "/sg-pretrain/checkpoints/expC_7b"
LOG_DIR = "/root/d_select/paper/experiments/logs"
TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"


# ════════════════════════════════════════════════════════════════
# Logger
# ════════════════════════════════════════════════════════════════

class Logger:
    def __init__(self, filepath, rank=0):
        self.rank = rank
        if rank == 0:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.f = open(filepath, "w")
        else:
            self.f = None

    def log(self, msg=""):
        if self.rank == 0:
            print(msg, flush=True)
            self.f.write(str(msg) + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


# ════════════════════════════════════════════════════════════════
# RoPE helpers
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
    """Build RoPE cos/sin cache for a given head_dim."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    emb = torch.outer(t, freqs)  # (seq_len, head_dim/2)
    cos = torch.cos(emb).to(dtype)
    sin = torch.sin(emb).to(dtype)
    # Stack to (seq_len, head_dim) for apply_rotary_pos_emb
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0)  # (1, seq_len, head_dim)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0)  # (1, seq_len, head_dim)
    return cos, sin


# ════════════════════════════════════════════════════════════════
# Thin Keys patching for LLaMA
# ════════════════════════════════════════════════════════════════

def patch_llama_thin_keys(model, d_select=1024):
    """Replace Q,K projections with smaller ones for thin keys.

    For each attention layer:
      - q_proj: (d_model, d_model) → (d_model, d_select)
      - k_proj: (d_model, d_model) → (d_model, d_select)
      - v_proj and o_proj: unchanged
      - RoPE uses d_head_qk = d_select // n_heads
      - SDPA handles asymmetric Q/K vs V dims
    """
    cfg = model.config
    d_model = cfg.hidden_size         # 4096
    n_heads = cfg.num_attention_heads  # 32
    d_head_qk = d_select // n_heads   # 32
    d_head_v = d_model // n_heads     # 128

    n_patched = 0
    for layer in model.model.layers:
        attn = layer.self_attn

        # Replace q_proj: (4096, 4096) → (4096, 1024)
        attn.q_proj = nn.Linear(d_model, d_select, bias=False)
        # Replace k_proj: (4096, 4096) → (4096, 1024)
        attn.k_proj = nn.Linear(d_model, d_select, bias=False)
        # v_proj and o_proj unchanged

        # Store thin config for custom forward
        attn._thin_cfg = dict(
            d_select=d_select, d_head_qk=d_head_qk,
            d_head_v=d_head_v, n_heads=n_heads, d_model=d_model,
        )

        # Replace forward to handle asymmetric dims + RoPE
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

                # Apply RoPE with thin head dim (d_head_qk)
                cos, sin = build_rope_cache(T, c['d_head_qk'], dev, dt)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                # SDPA handles asymmetric Q/K (d_head_qk) vs V (d_head_v)
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True,
                    scale=1.0 / math.sqrt(c['d_head_qk']),
                )
                # Output shape: (B, n_heads, T, d_head_v) → (B, T, d_model)
                out = out.transpose(1, 2).contiguous().view(B, T, c['d_model'])
                return (a.o_proj(out), None)
            return fwd

        attn.forward = make_forward(attn)
        n_patched += 1

    return n_patched


# ════════════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════════════

class BinDataset(Dataset):
    """Memory-mapped token dataset from binary uint16 file."""
    def __init__(self, path, seq_len, start_idx=0, end_idx=None):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        end_idx = end_idx or len(self.data)
        self.start = start_idx
        self.seq_len = seq_len
        self.n = (end_idx - start_idx - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        i = self.start + idx * self.seq_len
        chunk = self.data[i : i + self.seq_len + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1].copy()), torch.from_numpy(chunk[1:].copy())


class WikiTextDataset(Dataset):
    """WikiText-103 chunked into fixed-length sequences."""
    def __init__(self, token_ids, seq_len):
        # token_ids: 1D tensor of token IDs
        n = len(token_ids) // seq_len
        self.data = token_ids[:n * seq_len].view(n, seq_len)

    def __len__(self):
        return self.data.shape[0] - 1  # need next-token targets

    def __getitem__(self, idx):
        # Use overlapping: x = data[idx], y = shifted by 1
        start = idx * self.data.shape[1]
        # Actually, just use consecutive chunks with next-token prediction
        x = self.data[idx]
        # For causal LM, targets are shifted input
        # We'll return (input, target) where target = input shifted left
        # But since we chunk, we need seq_len+1 tokens per sample
        return x[:-1], x[1:]


def load_wikitext_validation(tokenizer, seq_len=1024):
    """Load WikiText-103 validation set, tokenize and chunk."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join([t for t in ds["text"] if t.strip()])
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # Need seq_len+1 per chunk for (input, target) pairs
    n_chunks = len(tokens) // (seq_len + 1)
    token_tensor = torch.tensor(tokens[:n_chunks * (seq_len + 1)], dtype=torch.long)
    token_tensor = token_tensor.view(n_chunks, seq_len + 1)

    class ChunkedDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return self.data.shape[0]
        def __getitem__(self, idx):
            return self.data[idx, :-1], self.data[idx, 1:]

    return ChunkedDataset(token_tensor)


def prepare_data(target_tokens=2_000_000_000):
    """Download OpenWebText via streaming and tokenize to binary file."""
    if os.path.exists(TOKEN_FILE):
        n = os.path.getsize(TOKEN_FILE) // 2
        print(f"Token file already exists: {TOKEN_FILE} ({n:,} tokens)")
        if n >= target_tokens:
            return
        print(f"  But only {n:,} < {target_tokens:,} target. Re-creating.")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    print(f"Preparing OpenWebText ({target_tokens / 1e9:.0f}B tokens)...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("openwebtext", split="train", streaming=True)

    all_ids = []
    t0 = time.time()
    for i, sample in enumerate(tqdm(ds, desc="Tokenizing", total=target_tokens // 500)):
        ids = tokenizer.encode(sample["text"])
        all_ids.extend(ids)
        if len(all_ids) >= target_tokens:
            break
        if (i + 1) % 200_000 == 0:
            elapsed = time.time() - t0
            rate = len(all_ids) / elapsed
            eta = (target_tokens - len(all_ids)) / max(rate, 1) / 3600
            print(f"  {len(all_ids) / 1e9:.2f}B tokens, "
                  f"{rate / 1e6:.1f}M tok/s, ETA {eta:.1f}h")

    all_ids = all_ids[:target_tokens]
    arr = np.array(all_ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    arr.tofile(TOKEN_FILE)
    elapsed = time.time() - t0
    print(f"Saved {len(arr):,} tokens to {TOKEN_FILE} "
          f"({os.path.getsize(TOKEN_FILE) / 1e9:.1f} GB, {elapsed / 60:.0f} min)")


# ════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════

def create_model(mode, d_select=1024, seed=42):
    """Create LLaMA-7B from random init, optionally patch with thin keys."""
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
        rope_theta=10000.0,
        tie_word_embeddings=False,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"

    torch.manual_seed(seed)
    model = LlamaForCausalLM(config)

    n_thin_params_saved = 0
    if mode == "thin_keys":
        # Count original Q,K params before patching
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
        print(f"  Patched {n} layers with thin keys (d_select={d_select})")
        print(f"  Q,K params: {orig_qk:,} → {new_qk:,} "
              f"(saved {n_thin_params_saved:,})")
    else:
        print(f"  Full attention (SDPA)")

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})

    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {n_total:,} ({n_total / 1e9:.2f}B)")

    return model, tokenizer, n_total, n_thin_params_saved


# ════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss, total_tok = 0.0, 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tok += y.numel()
    model.train()
    avg = total_loss / max(total_tok, 1)
    return avg, math.exp(min(avg, 20))  # cap to avoid overflow


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
    if args.seed != 42:
        tag = f"{tag}_seed{args.seed}"
    log_path = os.path.join(LOG_DIR, f"expC_7b_{tag}.log")
    logger = Logger(log_path, rank=rank)

    # ── Compute schedule ──
    eff_batch = world_size * args.batch_size * args.accumulation_steps
    tok_per_step = eff_batch * args.seq_len
    total_steps = args.total_tokens // tok_per_step

    logger.log("=" * 70)
    logger.log(f"EXPERIMENT C — FROM-SCRATCH 7B: {args.mode.upper()}")
    logger.log("=" * 70)
    logger.log(f"  GPUs: {world_size}")
    logger.log(f"  Batch: {args.batch_size}/gpu × {args.accumulation_steps} accum "
               f"× {world_size} gpus = {eff_batch}")
    logger.log(f"  Tokens/step: {tok_per_step:,}  |  Total steps: {total_steps:,}")
    logger.log(f"  Tokens: {args.total_tokens / 1e9:.1f}B  |  Seq len: {args.seq_len}")
    logger.log(f"  LR: {args.lr} → {args.lr / 10}  |  Warmup: {args.warmup_steps}")
    logger.log(f"  Seed: {args.seed}")
    if args.mode == "thin_keys":
        logger.log(f"  d_select: {args.d_select} (d_model/4)")

    # ── Data ──
    assert os.path.exists(TOKEN_FILE), f"Run --prepare_data first: {TOKEN_FILE}"
    n_tokens = os.path.getsize(TOKEN_FILE) // 2
    split = int(n_tokens * 0.95)

    train_ds = BinDataset(TOKEN_FILE, args.seq_len, 0, split)
    val_ds = BinDataset(TOKEN_FILE, args.seq_len, split, n_tokens)
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

    # ── Model (created on CPU, then FSDP moves to GPU) ──
    logger.log(f"\n  Creating {args.mode} model...")
    model, tokenizer, n_params, n_saved = create_model(
        args.mode, d_select=args.d_select, seed=args.seed)

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

    # ── Training loop ──
    os.makedirs(CKPT_DIR, exist_ok=True)
    best_ppl, best_step = float("inf"), 0
    trajectory = []
    step, micro = 0, 0
    t0 = time.time()
    pbar = tqdm(total=total_steps, desc=args.mode, disable=(rank != 0))

    model.train()
    epoch = 0
    opt.zero_grad()

    while step < total_steps:
        if train_sampler:
            train_sampler.set_epoch(epoch)

        for x, y in train_loader:
            if step >= total_steps:
                break

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
                        ppl=f"{best_ppl:.1f}")
                    pbar.update(1)

                # ── Eval ──
                if step % args.eval_interval == 0 or step >= total_steps:
                    # OWT validation
                    vl, vp = evaluate(model, val_loader, device,
                                      max_batches=args.eval_batches)

                    if dist.is_initialized():
                        t = torch.tensor([vl], device=device)
                        dist.all_reduce(t, op=dist.ReduceOp.AVG)
                        vl = t.item()
                        vp = math.exp(min(vl, 20))

                    # WikiText-103 validation
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
                        f"tok={toks / 1e9:.2f}B  t={hrs:.1f}h")
                    logger.log(log_msg)

                    entry = dict(
                        step=step, owt_val_ppl=vp, owt_val_loss=vl,
                        tokens=toks, hours=hrs)
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
                                f"expC_7b_{tag}_trajectory.json"),
                                "w") as f:
                            json.dump(trajectory, f, indent=2)

                    model.train()

        epoch += 1

    pbar.close()

    # ── Save checkpoint (gather FSDP shards → full state dict) ──
    if args.save_checkpoint:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            state_dict = model.state_dict()
        if rank == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"expC_7b_{tag}.pt")
            logger.log(f"\n  Saving checkpoint to {ckpt_path}...")
            torch.save(state_dict, ckpt_path)
            size_gb = os.path.getsize(ckpt_path) / 1e9
            logger.log(f"  Saved ({size_gb:.1f} GB)")
        if dist.is_initialized():
            dist.barrier()

    # ── Summary ──
    total_hrs = (time.time() - t0) / 3600
    logger.log(f"\n{'=' * 70}")
    logger.log(f"DONE: {args.mode.upper()}")
    logger.log(f"  Best OWT PPL: {best_ppl:.2f} at step {best_step}")
    if trajectory and "wt103_val_ppl" in trajectory[-1]:
        logger.log(f"  Final WT-103 PPL: {trajectory[-1]['wt103_val_ppl']:.2f}")
    logger.log(f"  Time: {total_hrs:.1f} hours")
    logger.log(f"  Tokens: {args.total_tokens / 1e9:.1f}B")
    logger.log(f"{'=' * 70}")

    if rank == 0:
        # Find best WT-103 PPL from trajectory
        best_wt103 = None
        if any("wt103_val_ppl" in e for e in trajectory):
            best_wt103 = min(e["wt103_val_ppl"] for e in trajectory
                             if "wt103_val_ppl" in e)

        results = {
            "experiment": "C",
            "mode": args.mode,
            "seed": args.seed,
            "best_owt_ppl": best_ppl,
            "best_owt_step": best_step,
            "best_wt103_ppl": best_wt103,
            "final_owt_ppl": trajectory[-1]["owt_val_ppl"] if trajectory else None,
            "final_wt103_ppl": trajectory[-1].get("wt103_val_ppl"),
            "total_hours": total_hrs,
            "total_tokens": args.total_tokens,
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

        with open(os.path.join(LOG_DIR, f"expC_7b_{tag}.json"), "w") as f:
            json.dump(results, f, indent=2)

    logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Experiment C: 7B from scratch — Full Attention vs Thin Keys")
    p.add_argument("--prepare_data", action="store_true")
    p.add_argument("--mode", choices=["full_attn", "thin_keys"],
                   default="full_attn")
    p.add_argument("--d_select", type=int, default=1024,
                   help="QK projection dim for thin_keys (d_model/4 = 1024)")
    p.add_argument("--total_tokens", type=int, default=2_000_000_000)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=2,
                   help="Per-GPU micro-batch size")
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=1000)
    p.add_argument("--eval_batches", type=int, default=50)
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for model init and data shuffling")
    p.add_argument("--save_checkpoint", action="store_true",
                   help="Save full model checkpoint at end of training")
    args = p.parse_args()

    if args.prepare_data:
        prepare_data(args.total_tokens)
        return

    train(args)


if __name__ == "__main__":
    main()

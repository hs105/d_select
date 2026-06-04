"""
Experiment F: Long-Context Evaluation for SVD-Compressed Mistral-7B
===================================================================
Addresses the #1 reviewer criticism: no long-context evaluation.

Three evaluations:
  1. Long-context PPL curves (4K -> 32K) using PG-19 books
  2. Needle-in-a-haystack (NIAH) retrieval at various depths & lengths
  3. Passkey retrieval (synthetic, exact-match)

Usage:
  # Full run: baseline + compressed
  python eval_longcontext.py --ranks 1024,512,256 --device cuda:0

  # Quick test (short contexts, few NIAH trials)
  python eval_longcontext.py --ranks 1024,256 --device cuda:0 \
      --ppl_lengths 4096,8192 --niah_lengths 4096,8192 --niah_depths 3 --quick

  # PPL only
  python eval_longcontext.py --ranks 1024,256 --eval ppl --device cuda:0

  # NIAH only
  python eval_longcontext.py --ranks 1024,256 --eval niah --device cuda:0

  # Use fine-tuned checkpoint instead of on-the-fly SVD
  python eval_longcontext.py --ranks 256 --checkpoint_dir /path/to/saved_model --device cuda:0
"""

import argparse
import gc
import json
import math
import os
import random
import time

import numpy as np
import torch


# ============================================================
# Compat: handle old transformers (4.22) that lack LlamaTokenizer
# ============================================================
def load_tokenizer(model_path):
    """Load tokenizer with fallback for old transformers versions."""
    # Try AutoTokenizer first (works on transformers >= 4.34)
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path)
    except (ValueError, ImportError) as e:
        if "LlamaTokenizer" not in str(e) and "does not exist" not in str(e):
            raise

    # Fallback: load from tokenizer.json directly (works on any version)
    print(f"  AutoTokenizer failed (old transformers?), trying fallback...")
    tokenizer_json = os.path.join(model_path, "tokenizer.json")
    if os.path.exists(tokenizer_json):
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
        # Load special tokens from tokenizer_config.json if available
        config_file = os.path.join(model_path, "tokenizer_config.json")
        if os.path.exists(config_file):
            with open(config_file) as f:
                tok_config = json.load(f)
            if "eos_token" in tok_config:
                eos = tok_config["eos_token"]
                if isinstance(eos, dict):
                    eos = eos.get("content", "</s>")
                tokenizer.eos_token = eos
            if "bos_token" in tok_config:
                bos = tok_config["bos_token"]
                if isinstance(bos, dict):
                    bos = bos.get("content", "<s>")
                tokenizer.bos_token = bos
        print(f"  Loaded tokenizer from {tokenizer_json} (vocab size: {len(tokenizer)})")
        return tokenizer

    raise RuntimeError(
        f"Cannot load tokenizer from {model_path}. "
        f"Need transformers >= 4.34 for AutoTokenizer, or tokenizer.json in model dir."
    )


def load_model(model_path, device):
    """Load model with fallback for old transformers versions."""
    from transformers import AutoModelForCausalLM
    # Try flash_attention_2 first, then sdpa, then default
    for attn_impl in ["flash_attention_2", "sdpa", None]:
        try:
            kwargs = dict(torch_dtype=torch.bfloat16)
            if attn_impl:
                kwargs["attn_implementation"] = attn_impl
            model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            return model.to(device)
        except (ImportError, ValueError, TypeError) as e:
            if attn_impl is None:
                raise
            continue
    raise RuntimeError(f"Failed to load model from {model_path}")


# ============================================================
# SVD Compression (shared with other experiment scripts)
# ============================================================
def compress_k_layers(model, rank, device='cuda:0', verbose=True):
    """SVD compress W_K in all layers to target rank."""
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    k_dim = n_kv_heads * d_head

    if rank >= k_dim:
        if verbose:
            print(f"  Rank {rank} >= K dim {k_dim}, skipping (baseline)")
        return []

    if verbose:
        print(f"  Compressing K to rank {rank} (saving {1-rank/k_dim:.0%})...")

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
    if verbose:
        print(f"  Done. Avg K error: {sum(errors)/len(errors):.4f}", flush=True)
    return errors


# ============================================================
# 1. Long-Context Perplexity
# ============================================================
def load_long_text(tokenizer, max_tokens=500_000, data_path=None):
    """Load long text for PPL evaluation. No internet needed if local files exist.

    Priority order:
      1. --data_path (user-supplied text file)
      2. Local WikiText-103 train set (already on machine from other experiments)
      3. Local WikiText-103 test set
      4. HuggingFace PG-19 download (only if above not found)
    """
    full_text = None

    # 1. User-supplied file
    if data_path and os.path.exists(data_path):
        print(f"  Loading from {data_path}...", flush=True)
        with open(data_path, 'r', encoding='utf-8') as f:
            full_text = f.read()

    # 2. Local WikiText-103 train (large, good for long-context eval)
    if full_text is None:
        for fpath in [
            '/root/data/wikitext-103/wiki.train.tokens',
            '/root/data/wikitext-103/wiki.test.tokens',
        ]:
            if os.path.exists(fpath):
                print(f"  Loading from {fpath}...", flush=True)
                with open(fpath, 'r', encoding='utf-8') as f:
                    full_text = f.read()
                break

    # 3. HuggingFace download (last resort)
    if full_text is None:
        try:
            from datasets import load_dataset
            print("  No local data found. Downloading PG-19 test set...", flush=True)
            ds = load_dataset("deepmind/pg19", split="test", trust_remote_code=True)
            texts = []
            total_tok = 0
            for item in ds:
                texts.append(item['text'])
                total_tok += len(item['text'].split())
                if total_tok > max_tokens * 2:
                    break
            full_text = "\n\n".join(texts)
        except Exception as e:
            raise RuntimeError(
                f"No text data found. Provide a text file via --data_path, "
                f"or place WikiText-103 at /root/data/wikitext-103/. "
                f"(download error: {e})"
            )

    print("  Tokenizing...", flush=True)
    tokens = tokenizer(full_text, return_tensors='pt')['input_ids'][0]
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    print(f"  Total tokens available: {len(tokens):,}", flush=True)
    return tokens


@torch.no_grad()
def eval_ppl_at_length(model, token_ids, seq_len, device, stride=None, max_eval_tokens=None):
    """Evaluate perplexity using a sliding window approach at a given context length.

    Uses stride-based evaluation (like huggingface PPL docs) to handle long sequences:
    the model sees `seq_len` tokens at a time, but only the last `stride` tokens
    contribute to the loss. This gives proper long-context PPL without padding artifacts.
    """
    if stride is None:
        stride = seq_len // 2  # 50% overlap by default

    n_tokens = len(token_ids)
    if max_eval_tokens and n_tokens > max_eval_tokens:
        n_tokens = max_eval_tokens

    nlls = []
    n_eval_tokens = 0

    for begin in range(0, n_tokens - 1, stride):
        end = min(begin + seq_len, n_tokens)
        input_ids = token_ids[begin:end].unsqueeze(0).to(device)

        # Only count loss on tokens after the overlap region
        target_start = 0 if begin == 0 else (seq_len - stride)
        target_ids = input_ids.clone()
        target_ids[0, :target_start] = -100  # mask overlap tokens

        outputs = model(input_ids, labels=target_ids)
        # outputs.loss is averaged over non-masked tokens
        n_valid = (target_ids != -100).sum().item() - 1  # -1 for shift
        # Recompute total NLL from the averaged loss
        # Actually, HF loss already handles the shift internally,
        # and averages over valid tokens. We need total NLL.
        nll = outputs.loss.item() * max(n_valid, 1)
        nlls.append(nll)
        n_eval_tokens += max(n_valid, 1)

        if end >= n_tokens:
            break

    avg_nll = sum(nlls) / max(n_eval_tokens, 1)
    ppl = math.exp(min(avg_nll, 20))
    return ppl, n_eval_tokens


def run_ppl_evaluation(model, tokenizer, device, context_lengths,
                       max_eval_tokens=200_000, data_path=None):
    """Run PPL evaluation at multiple context lengths."""
    print("\n" + "=" * 60)
    print("Long-Context Perplexity Evaluation")
    print("=" * 60)

    tokens = load_long_text(tokenizer, max_tokens=max_eval_tokens + 50_000,
                            data_path=data_path)

    results = {}
    for seq_len in context_lengths:
        if seq_len > len(tokens):
            print(f"  Skipping ctx={seq_len} (need {seq_len} tokens, have {len(tokens)})")
            continue

        print(f"\n  Context length: {seq_len:,}...", flush=True)
        t0 = time.time()

        try:
            ppl, n_tok = eval_ppl_at_length(
                model, tokens, seq_len, device,
                stride=seq_len // 2,
                max_eval_tokens=max_eval_tokens,
            )
            elapsed = time.time() - t0
            print(f"    PPL = {ppl:.2f} (evaluated {n_tok:,} tokens in {elapsed:.1f}s)")
            results[seq_len] = {
                'ppl': round(ppl, 2),
                'n_eval_tokens': n_tok,
                'elapsed_s': round(elapsed, 1),
            }
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM at ctx={seq_len}!")
            gc.collect()
            torch.cuda.empty_cache()
            results[seq_len] = {'error': 'OOM'}
            break

    return results


# ============================================================
# 2. Needle-in-a-Haystack (NIAH)
# ============================================================

NEEDLE_TEMPLATE = "The special magic number is {number}."

NIAH_QUERY = (
    "Based on the content of the text above, what is the special magic number? "
    "Answer with just the number, nothing else."
)

HAYSTACK_FILLER = (
    "The economy of the country grew at an annual rate of 3.2 percent in the last quarter, "
    "driven primarily by increased consumer spending and a rebound in manufacturing output. "
    "Government officials noted that inflation had remained within target levels despite "
    "rising energy costs. Analysts expect continued moderate growth through the remainder of "
    "the fiscal year, supported by favorable labor market conditions and steady export demand. "
    "Meanwhile, central bank policymakers signaled that interest rates would remain unchanged "
    "at their current level for at least two more quarters. "
)


def build_niah_prompt(tokenizer, context_length, depth_pct, needle_number):
    """Build a needle-in-a-haystack prompt.

    Args:
        context_length: Target total length in tokens
        depth_pct: Where to place the needle (0.0 = beginning, 1.0 = end)
        needle_number: The number to hide in the text
    """
    needle = NEEDLE_TEMPLATE.format(number=needle_number)

    # Tokenize filler to know its length
    filler_tokens = tokenizer(HAYSTACK_FILLER, add_special_tokens=False)['input_ids']
    needle_tokens = tokenizer(needle, add_special_tokens=False)['input_ids']
    query_tokens = tokenizer("\n\n" + NIAH_QUERY, add_special_tokens=False)['input_ids']

    # Reserve space for needle + query + some margin
    filler_budget = context_length - len(needle_tokens) - len(query_tokens) - 10

    # Build filler by repeating
    n_repeats = (filler_budget // len(filler_tokens)) + 1
    all_filler = filler_tokens * n_repeats
    all_filler = all_filler[:filler_budget]

    # Insert needle at depth
    insert_pos = int(len(all_filler) * depth_pct)
    # Align to sentence boundary (after a period token if possible)
    period_id = tokenizer.encode(".", add_special_tokens=False)[0]
    for offset in range(min(50, len(all_filler) - insert_pos)):
        if all_filler[insert_pos + offset] == period_id:
            insert_pos = insert_pos + offset + 1
            break

    prompt_tokens = (
        all_filler[:insert_pos]
        + needle_tokens
        + all_filler[insert_pos:]
        + query_tokens
    )

    # Trim to exact context length
    prompt_tokens = prompt_tokens[:context_length]

    return prompt_tokens


@torch.no_grad()
def run_niah_trial(model, tokenizer, device, context_length, depth_pct,
                   max_new_tokens=20):
    """Run a single NIAH trial. Returns (needle_number, model_answer, is_correct)."""
    needle_number = random.randint(1000000, 9999999)

    prompt_tokens = build_niah_prompt(tokenizer, context_length, depth_pct, needle_number)
    input_ids = torch.tensor([prompt_tokens], device=device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0, input_ids.shape[1]:]
    answer_text = tokenizer.decode(generated, skip_special_tokens=True).strip()

    is_correct = str(needle_number) in answer_text

    return needle_number, answer_text, is_correct


def run_niah_evaluation(model, tokenizer, device, context_lengths, n_depths=5,
                        n_trials_per=3):
    """Run full NIAH evaluation grid."""
    print("\n" + "=" * 60)
    print("Needle-in-a-Haystack Evaluation")
    print("=" * 60)

    depths = [i / (n_depths - 1) for i in range(n_depths)] if n_depths > 1 else [0.5]

    results = {}
    for ctx_len in context_lengths:
        print(f"\n  Context length: {ctx_len:,}")
        ctx_results = {}

        for depth in depths:
            depth_key = f"{depth:.2f}"
            correct = 0
            trials = []

            for trial in range(n_trials_per):
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    needle, answer, ok = run_niah_trial(
                        model, tokenizer, device, ctx_len, depth)
                    trials.append({
                        'needle': needle,
                        'answer': answer,
                        'correct': ok,
                    })
                    if ok:
                        correct += 1
                except torch.cuda.OutOfMemoryError:
                    print(f"    OOM at ctx={ctx_len}, depth={depth:.0%}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    trials.append({'error': 'OOM'})

            accuracy = correct / max(len([t for t in trials if 'error' not in t]), 1)
            ctx_results[depth_key] = {
                'accuracy': round(accuracy, 3),
                'correct': correct,
                'total': len(trials),
                'trials': trials,
            }
            status = "PASS" if accuracy >= 0.9 else ("PARTIAL" if accuracy > 0 else "FAIL")
            print(f"    depth={depth:.0%}: {correct}/{len(trials)} correct [{status}]")

        results[ctx_len] = ctx_results

    return results


# ============================================================
# 3. Passkey Retrieval
# ============================================================

PASSKEY_TEMPLATE = (
    "There is an important info hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will quiz you about the important information there.\n\n"
    "{filler_before}"
    "The pass key is {passkey}. Remember it. {passkey} is the pass key.\n"
    "{filler_after}"
    "\nWhat is the pass key? The pass key is"
)

PASSKEY_FILLER_SENTENCE = (
    "The grass is green. The sky is blue. The sun is yellow. "
    "Here we go. There and back again. "
)


def build_passkey_prompt(tokenizer, context_length, depth_pct, passkey):
    """Build a passkey retrieval prompt at target context length."""
    filler_tokens = tokenizer(PASSKEY_FILLER_SENTENCE, add_special_tokens=False)['input_ids']

    # Estimate how many filler tokens we need
    passkey_line = f"The pass key is {passkey}. Remember it. {passkey} is the pass key.\n"
    overhead_tokens = tokenizer(
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorize it. I will quiz you about the important information there.\n\n"
        + passkey_line
        + "\nWhat is the pass key? The pass key is",
        add_special_tokens=False
    )['input_ids']

    filler_budget = context_length - len(overhead_tokens) - 5
    n_repeats = (filler_budget // len(filler_tokens)) + 1
    all_filler = filler_tokens * n_repeats
    all_filler = all_filler[:filler_budget]

    split_pos = int(len(all_filler) * depth_pct)
    filler_before = tokenizer.decode(all_filler[:split_pos])
    filler_after = tokenizer.decode(all_filler[split_pos:])

    prompt = PASSKEY_TEMPLATE.format(
        filler_before=filler_before,
        filler_after=filler_after,
        passkey=passkey,
    )

    tokens = tokenizer(prompt, return_tensors='pt')['input_ids']
    return tokens


@torch.no_grad()
def run_passkey_trial(model, tokenizer, device, context_length, depth_pct,
                      max_new_tokens=10):
    """Run a single passkey retrieval trial."""
    passkey = random.randint(10000, 99999)
    input_ids = build_passkey_prompt(tokenizer, context_length, depth_pct, passkey)
    input_ids = input_ids[:, :context_length].to(device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = outputs[0, input_ids.shape[1]:]
    answer_text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    is_correct = str(passkey) in answer_text

    return passkey, answer_text, is_correct


def run_passkey_evaluation(model, tokenizer, device, context_lengths, n_depths=5,
                           n_trials_per=3):
    """Run passkey retrieval evaluation grid."""
    print("\n" + "=" * 60)
    print("Passkey Retrieval Evaluation")
    print("=" * 60)

    depths = [i / (n_depths - 1) for i in range(n_depths)] if n_depths > 1 else [0.5]

    results = {}
    for ctx_len in context_lengths:
        print(f"\n  Context length: {ctx_len:,}")
        ctx_results = {}

        for depth in depths:
            depth_key = f"{depth:.2f}"
            correct = 0
            trials = []

            for trial in range(n_trials_per):
                try:
                    gc.collect()
                    torch.cuda.empty_cache()
                    passkey, answer, ok = run_passkey_trial(
                        model, tokenizer, device, ctx_len, depth)
                    trials.append({
                        'passkey': passkey,
                        'answer': answer,
                        'correct': ok,
                    })
                    if ok:
                        correct += 1
                except torch.cuda.OutOfMemoryError:
                    print(f"    OOM at ctx={ctx_len}, depth={depth:.0%}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    trials.append({'error': 'OOM'})

            accuracy = correct / max(len([t for t in trials if 'error' not in t]), 1)
            ctx_results[depth_key] = {
                'accuracy': round(accuracy, 3),
                'correct': correct,
                'total': len(trials),
                'trials': trials,
            }
            status = "PASS" if accuracy >= 0.9 else ("PARTIAL" if accuracy > 0 else "FAIL")
            print(f"    depth={depth:.0%}: {correct}/{len(trials)} correct [{status}]")

        results[ctx_len] = ctx_results

    return results


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Long-context evaluation for SVD-compressed Mistral-7B")
    parser.add_argument('--model_path', type=str,
                        default='/sg-pretrain/models/mistral-7b')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Load a fine-tuned checkpoint instead of on-the-fly SVD')
    parser.add_argument('--ranks', type=str, default='1024,512,256',
                        help='Comma-separated ranks (1024=baseline)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--eval', type=str, default='all',
                        help='Which evals to run: all, ppl, niah, passkey')

    # PPL options
    parser.add_argument('--ppl_lengths', type=str, default='4096,8192,16384,32768',
                        help='Context lengths for PPL evaluation')
    parser.add_argument('--ppl_max_tokens', type=int, default=200_000,
                        help='Max tokens to evaluate for PPL')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to a text file for PPL eval (avoids any download). '
                             'If not set, uses local WikiText-103 at /root/data/wikitext-103/')

    # NIAH / passkey options
    parser.add_argument('--niah_lengths', type=str, default='4096,8192,16384,32768',
                        help='Context lengths for NIAH/passkey')
    parser.add_argument('--niah_depths', type=int, default=5,
                        help='Number of depth positions to test')
    parser.add_argument('--niah_trials', type=int, default=3,
                        help='Trials per (length, depth) cell')

    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: fewer tokens, fewer trials')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str,
                        default='/sg-pretrain/focus/paper/experiments/logs')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.quick:
        args.ppl_max_tokens = 50_000
        args.niah_trials = 1

    device = torch.device(args.device)
    ranks = [int(r) for r in args.ranks.split(',')]
    ppl_lengths = [int(x) for x in args.ppl_lengths.split(',')]
    niah_lengths = [int(x) for x in args.niah_lengths.split(',')]

    evals_to_run = args.eval.split(',') if ',' in args.eval else [args.eval]
    run_ppl = 'all' in evals_to_run or 'ppl' in evals_to_run
    run_niah = 'all' in evals_to_run or 'niah' in evals_to_run
    run_passkey = 'all' in evals_to_run or 'passkey' in evals_to_run

    print("=" * 70)
    print("Experiment F: Long-Context Evaluation")
    print("=" * 70)
    print(f"  Model: {args.model_path}")
    print(f"  Ranks: {ranks}")
    print(f"  Evals: ppl={run_ppl}, niah={run_niah}, passkey={run_passkey}")
    print(f"  PPL lengths: {ppl_lengths}")
    print(f"  NIAH lengths: {niah_lengths}")
    print(f"  NIAH depths: {args.niah_depths}, trials/cell: {args.niah_trials}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    tokenizer = load_tokenizer(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    for rank in ranks:
        tag = f"r{rank}" if rank < 1024 else "baseline"
        print(f"\n{'='*70}")
        print(f"Configuration: {tag} (rank={rank})")
        print(f"{'='*70}")

        # Load model
        if args.checkpoint_dir and rank < 1024:
            ckpt_path = os.path.join(args.checkpoint_dir, f"rank_{rank}")
            if os.path.exists(ckpt_path):
                print(f"\nLoading fine-tuned checkpoint from {ckpt_path}...", flush=True)
                model = load_model(ckpt_path, device)
            else:
                print(f"  Checkpoint {ckpt_path} not found, using on-the-fly SVD")
                model = None
        else:
            model = None

        if model is None:
            print(f"\nLoading model from {args.model_path}...", flush=True)
            t0 = time.time()
            model = load_model(args.model_path, device)
            print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

            if rank < 1024:
                print(f"\nApplying SVD compression (rank={rank})...", flush=True)
                compress_k_layers(model, rank, device=str(device))
                torch.cuda.empty_cache()

        model.eval()

        rank_results = {}

        # 1. PPL
        if run_ppl:
            try:
                ppl_results = run_ppl_evaluation(
                    model, tokenizer, device,
                    ppl_lengths, args.ppl_max_tokens, args.data_path)
                rank_results['ppl'] = ppl_results
            except Exception as e:
                print(f"  PPL evaluation failed: {e}")
                rank_results['ppl'] = {'error': str(e)}

        # 2. NIAH
        if run_niah:
            try:
                niah_results = run_niah_evaluation(
                    model, tokenizer, device,
                    niah_lengths, args.niah_depths, args.niah_trials)
                rank_results['niah'] = niah_results
            except Exception as e:
                print(f"  NIAH evaluation failed: {e}")
                rank_results['niah'] = {'error': str(e)}

        # 3. Passkey
        if run_passkey:
            try:
                passkey_results = run_passkey_evaluation(
                    model, tokenizer, device,
                    niah_lengths, args.niah_depths, args.niah_trials)
                rank_results['passkey'] = passkey_results
            except Exception as e:
                print(f"  Passkey evaluation failed: {e}")
                rank_results['passkey'] = {'error': str(e)}

        all_results[tag] = rank_results

        # Free model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ============================================================
    # Save results
    # ============================================================
    os.makedirs(args.save_dir, exist_ok=True)
    rank_str = "_".join(str(r) for r in ranks)
    save_path = os.path.join(args.save_dir, f'longcontext_{rank_str}.json')
    with open(save_path, 'w') as f:
        json.dump({
            'args': vars(args),
            'results': all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved results to {save_path}")

    # ============================================================
    # Print summary tables
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # PPL table
    if run_ppl:
        print(f"\n--- Long-Context Perplexity ---")
        header = f"{'Config':<12}"
        for ctx in ppl_lengths:
            header += f" {'ctx='+str(ctx//1024)+'K':>10}"
        print(header)
        print("-" * len(header))

        for tag in all_results:
            ppl_data = all_results[tag].get('ppl', {})
            if 'error' in ppl_data:
                continue
            row = f"{tag:<12}"
            for ctx in ppl_lengths:
                val = ppl_data.get(ctx, {})
                if isinstance(val, dict) and 'ppl' in val:
                    row += f" {val['ppl']:>10.2f}"
                elif isinstance(val, dict) and 'error' in val:
                    row += f" {'OOM':>10}"
                else:
                    row += f" {'--':>10}"
            print(row)

    # NIAH summary (average accuracy per context length)
    for eval_name in ['niah', 'passkey']:
        if eval_name == 'niah' and not run_niah:
            continue
        if eval_name == 'passkey' and not run_passkey:
            continue

        label = "Needle-in-a-Haystack" if eval_name == 'niah' else "Passkey Retrieval"
        print(f"\n--- {label} (avg accuracy) ---")
        header = f"{'Config':<12}"
        for ctx in niah_lengths:
            header += f" {'ctx='+str(ctx//1024)+'K':>10}"
        print(header)
        print("-" * len(header))

        for tag in all_results:
            data = all_results[tag].get(eval_name, {})
            if 'error' in data:
                continue
            row = f"{tag:<12}"
            for ctx in niah_lengths:
                ctx_data = data.get(ctx, {})
                if ctx_data:
                    accs = [v['accuracy'] for v in ctx_data.values()
                            if isinstance(v, dict) and 'accuracy' in v]
                    if accs:
                        avg_acc = sum(accs) / len(accs)
                        row += f" {avg_acc:>9.1%}"
                    else:
                        row += f" {'--':>10}"
                else:
                    row += f" {'--':>10}"
            print(row)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()

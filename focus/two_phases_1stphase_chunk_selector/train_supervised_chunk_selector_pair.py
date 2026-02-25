"""
Score all C-choose-k pairs of chunks, then aggregate to per-chunk scores.
Tests whether pair-based scoring recovers correct chunk rankings.
"""
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from itertools import combinations
from tqdm import tqdm
from datetime import datetime
import os


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    T: int = 96
    C: int = 12
    k: int = 2
    L: int = 8
    model_name: str = "Qwen/Qwen2.5-3B"
    log_dir: str = "/root/data/logs"


def simple_word_tokenize(text):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())

def build_vocab(all_tokens_list):
    vocab = {"<pad>": 0, "<unk>": 1}
    for tokens in all_tokens_list:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab

def encode(tokens, vocab):
    return [vocab.get(t, vocab["<unk>"]) for t in tokens]

def prepare_sentence(tokens, vocab, T, pad_id):
    ids = encode(tokens, vocab)
    if len(ids) < T:
        ids = ids + [pad_id] * (T - len(ids))
    else:
        ids = ids[:T]
    return torch.tensor(ids, dtype=torch.long)

def find_answer_chunks(tokens, answer_lower, L):
    answer_chunks = []
    for chunk_idx in range(len(tokens) // L + 1):
        start = chunk_idx * L
        end = min(start + L, len(tokens))
        chunk_toks = tokens[start:end]
        if answer_lower in chunk_toks:
            answer_chunks.append(chunk_idx)
    return answer_chunks


@torch.no_grad()
def score_full_sequence(lm_model, lm_tokenizer, context_texts, target_texts, device):
    """Score full answer sequence: sum of logP for all answer tokens."""
    logprobs = []
    mini_batch_size = 4
    
    for i in range(0, len(context_texts), mini_batch_size):
        batch_ctx = context_texts[i:i+mini_batch_size]
        batch_tgt = target_texts[i:i+mini_batch_size]
        
        combined = [ctx + tgt for ctx, tgt in zip(batch_ctx, batch_tgt)]
        
        ctx_inputs = lm_tokenizer(batch_ctx, return_tensors="pt", padding=True, truncation=True).to(device)
        full_inputs = lm_tokenizer(combined, return_tensors="pt", padding=True, truncation=True).to(device)
        
        outputs = lm_model(**full_inputs)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        
        ctx_lens = ctx_inputs.attention_mask.sum(dim=1)
        full_lens = full_inputs.attention_mask.sum(dim=1)
        
        for j in range(len(batch_ctx)):
            ctx_len = ctx_lens[j].item()
            full_len = full_lens[j].item()
            
            if full_len <= ctx_len:
                logprobs.append(-100.0)
                continue
            
            total_logp = 0.0
            n_tokens = 0
            for pos in range(ctx_len - 1, full_len - 1):
                next_token_id = full_inputs.input_ids[j, pos + 1]
                if full_inputs.attention_mask[j, pos + 1] == 0:
                    break
                total_logp += log_probs[j, pos, next_token_id].item()
                n_tokens += 1
            
            logprobs.append(total_logp if n_tokens > 0 else -100.0)
    
    return torch.tensor(logprobs, device=device)


def score_all_pairs(lm_model, lm_tokenizer, tokens_list, questions, answers, cfg, device):
    """
    For each example, score all C-choose-k pairs of chunks.
    Returns per-chunk aggregated scores: [N, C]
    """
    L = cfg.L
    C = cfg.C
    k = cfg.k
    N = len(tokens_list)
    
    all_chunk_scores = []
    
    # Baseline: logP(answer | question) for each example
    baseline_logps = score_full_sequence(lm_model, lm_tokenizer, questions, answers, device)
    
    for ex_idx in tqdm(range(N), desc="Scoring pairs"):
        tokens = tokens_list[ex_idx]
        question = questions[ex_idx]
        answer = answers[ex_idx]
        baseline = baseline_logps[ex_idx].item()
        
        # Extract chunk texts
        chunk_texts = []
        valid_chunks = []
        for c in range(C):
            start = c * L
            end = min(start + L, len(tokens))
            chunk_toks = tokens[start:end] if start < len(tokens) else []
            chunk_toks = [t for t in chunk_toks if t != "<pad>"]
            if len(chunk_toks) == 0:
                valid_chunks.append(False)
                chunk_texts.append("")
            else:
                valid_chunks.append(True)
                chunk_texts.append(" ".join(chunk_toks))
        
        # Get all valid chunk indices
        valid_indices = [c for c in range(C) if valid_chunks[c]]
        
        # Generate all pairs of valid chunks
        pairs = list(combinations(valid_indices, k))
        
        if len(pairs) == 0:
            all_chunk_scores.append(torch.zeros(C, device=device))
            continue
        
        # Build contexts for all pairs
        pair_contexts = []
        for pair in pairs:
            pair_text = " ".join([chunk_texts[c] for c in sorted(pair)])
            pair_contexts.append(pair_text + " " + question)
        pair_targets = [answer] * len(pairs)
        
        # Score all pairs
        pair_logps = score_full_sequence(lm_model, lm_tokenizer, pair_contexts, pair_targets, device)
        pair_rewards = pair_logps - baseline  # contrastive
        
        # Aggregate: for each chunk, average reward of all pairs it appears in
        chunk_score_sum = torch.zeros(C, device=device)
        chunk_count = torch.zeros(C, device=device)
        
        for pair_idx, pair in enumerate(pairs):
            for c in pair:
                chunk_score_sum[c] += pair_rewards[pair_idx].item()
                chunk_count[c] += 1
        
        chunk_scores = torch.where(
            chunk_count > 0,
            chunk_score_sum / chunk_count,
            torch.tensor(-100.0, device=device)
        )
        
        all_chunk_scores.append(chunk_scores)
    
    return torch.stack(all_chunk_scores)


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    
    with open('/root/data/train_data.json', 'r') as f:
        train_data = json.load(f)
    with open('/root/data/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    # Build vocab
    all_texts = []
    for ex in train_data + test_data:
        combined = " ".join(ex["sentences"])
        all_texts.append(simple_word_tokenize(combined))
    vocab = build_vocab(all_texts)
    pad_id = vocab["<pad>"]
    
    # Prepare test data
    test_tokens = []
    test_questions = []
    test_answers = []
    test_gt = []
    test_names = []
    
    for ex in test_data:
        combined = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined)
        test_tokens.append(tokens)
        test_questions.append(ex["question"])
        test_answers.append(" " + ex["answer"])
        test_gt.append(find_answer_chunks(tokens, ex["answer_lower"], cfg.L))
        test_names.append(ex["answer"])
    
    # Load LM
    print("Loading Qwen...")
    lm_tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    lm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    lm_model.eval()
    
    # === Individual chunk scoring (baseline, reproduced) ===
    print("\n=== Individual Chunk Scoring ===")
    baseline_logps = score_full_sequence(lm_model, lm_tokenizer, test_questions, test_answers, device)
    
    indiv_correct = 0
    for i in range(len(test_data)):
        tokens = test_tokens[i]
        question = test_questions[i]
        answer = test_answers[i]
        bl = baseline_logps[i].item()
        
        chunk_contexts = []
        valid = []
        for c in range(cfg.C):
            start = c * cfg.L
            end = min(start + cfg.L, len(tokens))
            chunk_toks = tokens[start:end] if start < len(tokens) else []
            chunk_toks = [t for t in chunk_toks if t != "<pad>"]
            if len(chunk_toks) == 0:
                valid.append(False)
                chunk_contexts.append(question)
            else:
                valid.append(True)
                chunk_contexts.append(" ".join(chunk_toks) + " " + question)
        
        chunk_logps = score_full_sequence(lm_model, lm_tokenizer, chunk_contexts, [answer]*cfg.C, device)
        scores = chunk_logps - bl
        for c in range(cfg.C):
            if not valid[c]:
                scores[c] = -999
        
        top_k = sorted(torch.topk(scores, k=cfg.k).indices.tolist())
        gt = test_gt[i]
        hit = len(set(top_k) & set(gt)) == len(gt) and len(gt) > 0
        if hit:
            indiv_correct += 1
        status = "✓" if hit else "✗"
        score_str = " ".join([f"{s:.2f}" if valid[c] else "----" for c, s in enumerate(scores)])
        print(f"  {i+1} {status}: {test_names[i]:15s} | Indiv top-2={top_k}, GT={gt} | [{score_str}]")
    
    print(f"\nIndividual scoring: {indiv_correct}/{len(test_data)} ({100*indiv_correct/len(test_data):.1f}%)")
    
    # === Pair-based chunk scoring ===
    print("\n=== Pair-Based Chunk Scoring ===")
    pair_scores = score_all_pairs(lm_model, lm_tokenizer, test_tokens, test_questions, test_answers, cfg, device)
    
    pair_correct = 0
    for i in range(len(test_data)):
        scores = pair_scores[i]
        scores_display = scores.clone()
        valid = scores > -50
        scores_masked = scores.clone()
        scores_masked[~valid] = -999
        
        top_k = sorted(torch.topk(scores_masked, k=cfg.k).indices.tolist())
        gt = test_gt[i]
        hit = len(set(top_k) & set(gt)) == len(gt) and len(gt) > 0
        if hit:
            pair_correct += 1
        status = "✓" if hit else "✗"
        score_str = " ".join([f"{s:.2f}" if valid[c] else "----" for c, s in enumerate(scores)])
        print(f"  {i+1} {status}: {test_names[i]:15s} | Pair  top-2={top_k}, GT={gt} | [{score_str}]")
    
    print(f"\nPair-based scoring: {pair_correct}/{len(test_data)} ({100*pair_correct/len(test_data):.1f}%)")
    
    # === Summary ===
    print(f"\n{'='*60}")
    print(f"Individual chunk scoring: {indiv_correct}/{len(test_data)} ({100*indiv_correct/len(test_data):.1f}%)")
    print(f"Pair-based chunk scoring: {pair_correct}/{len(test_data)} ({100*pair_correct/len(test_data):.1f}%)")
    print(f"REINFORCE (from earlier):  8/10 (80.0%)")
    print(f"Oracle string-match:      10/10 (100.0%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
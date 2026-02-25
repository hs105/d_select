"""
Score each chunk independently: logP(answer | chunk_i + question) - logP(answer | question)
Then train policy with supervised learning on these scores.
No REINFORCE needed.
"""
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from tqdm import tqdm
from datetime import datetime
import os
import random


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    T: int = 96
    C: int = 12
    k: int = 2
    L: int = 8

    train_steps: int = 2000
    batch_size: int = 8
    lr: float = 1e-3
    warmup_steps: int = 200

    model_name: str = "Qwen/Qwen2.5-3B"
    
    log_dir: str = "/root/data/logs"
    log_every: int = 100
    eval_every: int = 500


class Logger:
    def __init__(self, log_dir, experiment_name):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}_metrics.json")
        self.metrics = {"config": {}, "training": [], "evaluations": [], "test_results": []}
        self.log(f"Experiment: {experiment_name}")
    
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_config(self, config):
        for key, value in vars(config).items():
            self.metrics["config"][key] = str(value)
    
    def log_eval(self, step, train_acc, test_acc):
        self.metrics["evaluations"].append({"step": step, "train_accuracy": train_acc, "test_accuracy": test_acc})
        self.log(f"Eval @ {step}: Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


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


@torch.no_grad()
def score_all_chunks(lm_model, lm_tokenizer, tokens_list, questions, answers, vocab_inv, cfg, device):
    """
    For each example, score every chunk independently:
    score[i] = logP(answer | chunk_i + question) - logP(answer | question)
    
    Returns: [N, C] tensor of per-chunk scores
    """
    N = len(tokens_list)
    C = cfg.C
    L = cfg.L
    all_scores = []
    
    # First: compute baseline logP(answer | question) for each example
    baseline_logps = score_full_sequence(lm_model, lm_tokenizer, questions, answers, device)
    
    for ex_idx in tqdm(range(N), desc="Scoring chunks"):
        tokens = tokens_list[ex_idx]
        question = questions[ex_idx]
        answer = answers[ex_idx]
        baseline = baseline_logps[ex_idx].item()
        
        # Build context for each chunk
        chunk_contexts = []
        valid_chunks = []
        
        for c in range(C):
            start = c * L
            end = min(start + L, len(tokens))
            chunk_toks = tokens[start:end] if start < len(tokens) else []
            chunk_toks = [t for t in chunk_toks if t != "<pad>"]
            
            if len(chunk_toks) == 0:
                valid_chunks.append(False)
                chunk_contexts.append(question)  # placeholder
            else:
                valid_chunks.append(True)
                chunk_contexts.append(" ".join(chunk_toks) + " " + question)
        
        # Score all chunks for this example
        chunk_targets = [answer] * C
        chunk_logps = score_full_sequence(lm_model, lm_tokenizer, chunk_contexts, chunk_targets, device)
        
        # Contrastive: subtract baseline
        scores = chunk_logps - baseline
        
        # Set invalid chunks to large negative
        for c in range(C):
            if not valid_chunks[c]:
                scores[c] = -100.0
        
        all_scores.append(scores)
    
    return torch.stack(all_scores)  # [N, C]


class ChunkPolicy(nn.Module):
    def __init__(self, vocab_size, C, L, d_model=128, pad_id=0):
        super().__init__()
        self.C = C
        self.L = L
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        B, T = x.shape
        xc = x.view(B, self.C, self.L)
        e = self.emb(xc)
        mask = (xc != self.pad_id).float()
        denom = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        h = (e * mask.unsqueeze(-1)).sum(dim=2) / denom
        logits = self.mlp(h).squeeze(-1)
        return logits


def get_valid_chunk_mask(x, pad_id, C, L):
    B, T = x.shape
    xc = x.view(B, C, L)
    return (xc != pad_id).any(dim=2)


def prepare_dataset(data, vocab, cfg, device):
    examples = []
    for ex in data:
        combined_text = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined_text)
        pad_id = vocab["<pad>"]
        x_input = prepare_sentence(tokens, vocab, cfg.T, pad_id).to(device)
        answer_chunks = find_answer_chunks(tokens, ex["answer_lower"], cfg.L)
        examples.append({
            "x": x_input,
            "tokens": tokens,
            "question": ex["question"],
            "answer": " " + ex["answer"],
            "answer_chunks": answer_chunks,
            "raw": ex,
        })
    return examples


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    
    logger = Logger(cfg.log_dir, "supervised_chunk_scores")
    logger.log_config(cfg)
    
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
    vocab_inv = {v: k for k, v in vocab.items()}
    pad_id = vocab["<pad>"]
    
    # Prepare examples
    train_examples = prepare_dataset(train_data, vocab, cfg, device)
    test_examples = prepare_dataset(test_data, vocab, cfg, device)
    logger.log(f"Train: {len(train_examples)}, Test: {len(test_examples)}")
    
    # Load LM for scoring
    logger.log("Loading Qwen model...")
    lm_tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    lm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    lm_model.eval()
    
    # === STEP 1: Score all chunks with LM (one-time cost) ===
    logger.log("\nScoring all chunks with LM (one-time)...")
    
    train_scores = score_all_chunks(
        lm_model, lm_tokenizer,
        [ex["tokens"] for ex in train_examples],
        [ex["question"] for ex in train_examples],
        [ex["answer"] for ex in train_examples],
        vocab_inv, cfg, device
    )
    
    test_scores = score_all_chunks(
        lm_model, lm_tokenizer,
        [ex["tokens"] for ex in test_examples],
        [ex["question"] for ex in test_examples],
        [ex["answer"] for ex in test_examples],
        vocab_inv, cfg, device
    )
    
    # Show LM chunk scores vs ground truth
    logger.log("\n=== LM Chunk Scores (test set) ===")
    lm_correct = 0
    for i, ex in enumerate(test_examples):
        scores = test_scores[i]
        valid = get_valid_chunk_mask(ex["x"].unsqueeze(0), pad_id, cfg.C, cfg.L)[0]
        scores_masked = scores.clone()
        scores_masked[~valid] = -999
        top_k = torch.topk(scores_masked, k=cfg.k).indices.tolist()
        top_k_sorted = sorted(top_k)
        gt = ex["answer_chunks"]
        overlap = len(set(top_k_sorted) & set(gt))
        hit = overlap == len(gt) and len(gt) > 0
        if hit:
            lm_correct += 1
        status = "✓" if hit else "✗"
        
        # Show all chunk scores
        score_str = " ".join([f"{s:.2f}" if valid[j] else "----" for j, s in enumerate(scores)])
        logger.log(f"  {i+1} {status}: {ex['raw']['answer']:15s} | LM top-{cfg.k}={top_k_sorted}, GT={gt} | Scores: [{score_str}]")
    
    logger.log(f"\nLM direct ranking accuracy: {lm_correct}/{len(test_examples)} ({100*lm_correct/len(test_examples):.1f}%)")
    
    # === STEP 2: Train policy to predict these scores (supervised) ===
    logger.log("\n=== Training policy with supervised learning ===")
    
    # Normalize scores to targets for MSE
    # Use softmax over valid chunks as target distribution
    train_targets = []
    for i in range(len(train_examples)):
        scores = train_scores[i]
        valid = get_valid_chunk_mask(train_examples[i]["x"].unsqueeze(0), pad_id, cfg.C, cfg.L)[0]
        # Mask invalid, then use scores directly as regression targets
        targets = scores.clone()
        targets[~valid] = 0.0
        train_targets.append(targets)
    train_targets = torch.stack(train_targets)  # [N_train, C]
    
    test_targets = []
    for i in range(len(test_examples)):
        scores = test_scores[i]
        valid = get_valid_chunk_mask(test_examples[i]["x"].unsqueeze(0), pad_id, cfg.C, cfg.L)[0]
        targets = scores.clone()
        targets[~valid] = 0.0
        test_targets.append(targets)
    test_targets = torch.stack(test_targets)
    
    # Free LM memory
    del lm_model
    torch.cuda.empty_cache()
    
    policy = ChunkPolicy(vocab_size=len(vocab), C=cfg.C, L=cfg.L, pad_id=pad_id).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    
    best_test_acc = 0.0
    best_state = None
    
    for step in tqdm(range(cfg.train_steps), desc="Training"):
        # LR warmup
        lr = cfg.lr * min(1.0, (step + 1) / cfg.warmup_steps)
        for pg in opt.param_groups:
            pg['lr'] = lr
        
        # Sample batch
        idx = random.sample(range(len(train_examples)), min(cfg.batch_size, len(train_examples)))
        x_batch = torch.stack([train_examples[i]["x"] for i in idx])
        target_batch = train_targets[idx]
        valid_batch = get_valid_chunk_mask(x_batch, pad_id, cfg.C, cfg.L)
        
        policy.train()
        pred = policy(x_batch)  # [B, C]
        
        # MSE loss on valid chunks only
        mask = valid_batch.float()
        loss = ((pred - target_batch) ** 2 * mask).sum() / mask.sum()
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        
        if (step + 1) % cfg.log_every == 0:
            tqdm.write(f"Step {step+1}: lr={lr:.2e}, loss={loss.item():.4f}")
        
        if (step + 1) % cfg.eval_every == 0:
            policy.eval()
            # Eval on train
            train_correct = 0
            for i, ex in enumerate(train_examples):
                with torch.no_grad():
                    logits = policy(ex["x"].unsqueeze(0))
                    valid = get_valid_chunk_mask(ex["x"].unsqueeze(0), pad_id, cfg.C, cfg.L)[0]
                    logits[0][~valid] = -999
                    top_k = torch.topk(logits[0], k=cfg.k).indices.tolist()
                    gt = set(ex["answer_chunks"])
                    if len(set(top_k) & gt) == len(gt):
                        train_correct += 1
            train_acc = 100 * train_correct / len(train_examples)
            
            # Eval on test
            test_correct = 0
            for i, ex in enumerate(test_examples):
                with torch.no_grad():
                    logits = policy(ex["x"].unsqueeze(0))
                    valid = get_valid_chunk_mask(ex["x"].unsqueeze(0), pad_id, cfg.C, cfg.L)[0]
                    logits[0][~valid] = -999
                    top_k = torch.topk(logits[0], k=cfg.k).indices.tolist()
                    gt = set(ex["answer_chunks"])
                    if len(set(top_k) & gt) == len(gt):
                        test_correct += 1
            test_acc = 100 * test_correct / len(test_examples)
            
            logger.log_eval(step + 1, train_acc, test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                logger.log(f"  ** New best: {test_acc:.1f}% **")
    
    # Restore best
    if best_state is not None:
        policy.load_state_dict(best_state)
        logger.log(f"\nRestored best checkpoint (test={best_test_acc:.1f}%)")
    
    # Final eval
    logger.log("\n=== Final Test Results ===")
    policy.eval()
    final_correct = 0
    for i, ex in enumerate(test_examples):
        with torch.no_grad():
            logits = policy(ex["x"].unsqueeze(0))
            valid = get_valid_chunk_mask(ex["x"].unsqueeze(0), pad_id, cfg.C, cfg.L)[0]
            logits[0][~valid] = -999
            top_k = sorted(torch.topk(logits[0], k=cfg.k).indices.tolist())
            gt = ex["answer_chunks"]
            overlap = len(set(top_k) & set(gt))
            hit = overlap == len(gt) and len(gt) > 0
            if hit:
                final_correct += 1
            status = "✓" if hit else "✗"
            logger.log(f"  {i+1} {status}: {ex['raw']['answer']:15s} | Policy={top_k}, GT={gt}")
    
    logger.log(f"\nFinal: {final_correct}/{len(test_examples)} ({100*final_correct/len(test_examples):.1f}%)")
    logger.save_metrics()


if __name__ == "__main__":
    main()
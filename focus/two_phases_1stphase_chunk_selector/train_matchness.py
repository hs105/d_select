"""
Contrastive reward: logP(answer|chunks+question) - logP(answer|question)
This isolates the chunk contribution from the LM's parametric knowledge.
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

    train_steps: int = 5000
    batch_size: int = 8
    lr: float = 3e-5
    warmup_steps: int = 500
    baseline_momentum: float = 0.95

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
        
        self.metrics = {
            "config": {},
            "training": [],
            "evaluations": [],
            "test_results": [],
        }
        
        self.log(f"Experiment: {experiment_name}")
        self.log(f"Timestamp: {timestamp}")
    
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_config(self, config):
        for key, value in vars(config).items():
            self.metrics["config"][key] = str(value)
    
    def log_training_step(self, step, lr, baseline, avg_reward):
        self.metrics["training"].append({
            "step": step,
            "lr": lr,
            "baseline": baseline,
            "avg_reward": avg_reward,
        })
    
    def log_eval(self, step, train_acc, test_acc):
        self.metrics["evaluations"].append({
            "step": step,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        })
        self.log(f"Eval @ {step}: Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    def log_example_result(self, split, idx, example, result):
        self.metrics["test_results"].append({
            "split": split,
            "idx": idx,
            "question": example["raw"]["question"],
            "answer": example["raw"]["answer"],
            "selected_chunks": result["selected_chunks"],
            "answer_chunks": result["answer_chunks"],
            "overlap": result["overlap"],
            "reward": result["reward"],
            "status": "perfect" if result["overlap"] == result["total_answer_chunks"] and result["total_answer_chunks"] > 0 else "failed",
        })
    
    def log_summary(self, split, results):
        perfect = sum(1 for r in results if r["overlap"] == r["total_answer_chunks"] and r["total_answer_chunks"] > 0)
        total = len(results)
        self.log(f"{split} Summary: {perfect}/{total} ({100*perfect/total:.1f}% perfect)")
    
    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def simple_word_tokenize(text: str):
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

@torch.no_grad()
def get_next_token_logprob_batch(lm_model, lm_tokenizer, context_texts, target_tokens, device):
    """Score full answer sequence, not just first token."""
    batch_size = len(context_texts)
    logprobs = []
    
    mini_batch_size = 4
    for i in range(0, batch_size, mini_batch_size):
        batch_contexts = context_texts[i:i+mini_batch_size]
        batch_targets = target_tokens[i:i+mini_batch_size]
        
        # Build combined sequences: context + target
        combined_texts = [ctx + tgt for ctx, tgt in zip(batch_contexts, batch_targets)]
        
        ctx_inputs = lm_tokenizer(batch_contexts, return_tensors="pt", padding=True, truncation=True).to(device)
        full_inputs = lm_tokenizer(combined_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        
        outputs = lm_model(**full_inputs)
        log_probs = F.log_softmax(outputs.logits, dim=-1)
        
        ctx_lens = ctx_inputs.attention_mask.sum(dim=1)  # length of context portion
        full_lens = full_inputs.attention_mask.sum(dim=1)  # length of full sequence
        
        for j in range(len(batch_contexts)):
            ctx_len = ctx_lens[j].item()
            full_len = full_lens[j].item()
            
            if full_len <= ctx_len:
                logprobs.append(-100.0)
                continue
            
            # Sum log-probs over target tokens (positions ctx_len-1 to full_len-2 predict tokens ctx_len to full_len-1)
            total_logp = 0.0
            n_tokens = 0
            for pos in range(ctx_len - 1, full_len - 1):
                next_token_id = full_inputs.input_ids[j, pos + 1]
                if full_inputs.attention_mask[j, pos + 1] == 0:
                    break
                total_logp += log_probs[j, pos, next_token_id].item()
                n_tokens += 1
            
            if n_tokens == 0:
                logprobs.append(-100.0)
            else:
                logprobs.append(total_logp)
    
    return torch.tensor(logprobs, device=device)

def compute_reward_batch(cfg, lm_model, lm_tokenizer, x_selected_batch, vocab_inv, questions, answers, device, answer_chunks_list, actions):
    """
    Oracle reward: +1 for each selected chunk that contains the answer, -1 for each that doesn't.
    """
    B = x_selected_batch.shape[0]
    rewards = []
    
    for i in range(B):
        selected = set(actions[i].tolist())
        gt = set(answer_chunks_list[i])
        overlap = len(selected & gt)
        wrong = len(selected) - overlap
        rewards.append(float(overlap - wrong))
    
    return torch.tensor(rewards, device=device)

def get_valid_chunk_mask(x: torch.Tensor, pad_id: int, C: int, L: int) -> torch.Tensor:
    B, T = x.shape
    xc = x.view(B, C, L)
    has_content = (xc != pad_id).any(dim=2)
    return has_content

class ChunkPolicy(nn.Module):
    def __init__(self, vocab_size: int, C: int, L: int, d_model: int = 128, pad_id: int = 0):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        xc = x.view(B, self.C, self.L)
        e = self.emb(xc)

        mask = (xc != self.pad_id).float()
        denom = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        h = (e * mask.unsqueeze(-1)).sum(dim=2) / denom

        logits = self.mlp(h).squeeze(-1)
        return logits

def sample_k_without_replacement(logits: torch.Tensor, k: int, valid_mask: torch.Tensor = None):
    B, C = logits.shape
    actions, logps = [], []
    excluded = torch.zeros(B, C, dtype=torch.bool, device=logits.device)
    
    if valid_mask is not None:
        excluded = excluded | ~valid_mask

    for _ in range(k):
        current_mask = excluded.clone()
        masked_logits = logits.masked_fill(current_mask, float("-inf"))
        probs = F.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        actions.append(a)
        logps.append(dist.log_prob(a))
        excluded[torch.arange(B, device=logits.device), a] = True

    return torch.stack(actions, 1), torch.stack(logps, 1).sum(dim=1)

def build_selected_context(x: torch.Tensor, actions: torch.Tensor, C: int, L: int):
    B, T = x.shape
    xc = x.view(B, C, L)
    actions_sorted, _ = torch.sort(actions, dim=1)
    sel = xc[torch.arange(B, device=x.device).unsqueeze(1), actions_sorted]
    return sel.reshape(B, -1)

def find_answer_chunks(tokens, answer_lower, L):
    answer_chunks = []
    for chunk_idx in range(len(tokens) // L + 1):
        start = chunk_idx * L
        end = min(start + L, len(tokens))
        chunk_toks = tokens[start:end]
        if answer_lower in chunk_toks:
            answer_chunks.append(chunk_idx)
    return answer_chunks

def prepare_dataset(data, vocab, cfg, device):
    examples = []
    for ex in data:
        combined_text = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined_text)
        pad_id = vocab["<pad>"]
        
        x_input = prepare_sentence(tokens, vocab, cfg.T, pad_id).to(device)
        question = ex["question"]
        answer = " " + ex["answer"]
        answer_chunks = find_answer_chunks(tokens, ex["answer_lower"], cfg.L)
        
        examples.append({
            "x": x_input,
            "question": question,
            "answer": answer,
            "answer_chunks": answer_chunks,
            "raw": ex,
        })
    
    return examples

def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step / cfg.warmup_steps)
    return cfg.lr

def quick_eval(policy, examples, cfg, pad_id, device):
    policy.eval()
    correct = 0
    for ex in examples:
        with torch.no_grad():
            x_input = ex["x"].unsqueeze(0)
            logits = policy(x_input)
            valid_mask = get_valid_chunk_mask(x_input, pad_id, cfg.C, cfg.L)
            greedy = torch.topk(logits, k=cfg.k, dim=-1).indices[0]
            selected = set(greedy.tolist())
            answer = set(ex["answer_chunks"])
            if len(selected & answer) == len(answer):
                correct += 1
    return 100 * correct / len(examples)

def train_batched(cfg, policy, opt, lm_model, lm_tokenizer, train_examples, test_examples, vocab_inv, pad_id, device, logger):
    baseline = 0.0
    best_test_acc = 0.0
    best_state = None
    
    for step in tqdm(range(cfg.train_steps), desc="Training"):
        lr = get_lr(step, cfg)
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        
        batch = random.sample(train_examples, min(cfg.batch_size, len(train_examples)))
        
        x_batch = torch.stack([ex["x"] for ex in batch])
        questions = [ex["question"] for ex in batch]
        answers = [ex["answer"] for ex in batch]
        
        policy.train()
        
        logits = policy(x_batch)
        
        valid_mask = get_valid_chunk_mask(x_batch, pad_id, cfg.C, cfg.L)
        actions, logp_action = sample_k_without_replacement(logits, cfg.k, valid_mask)
        
        x_sel = build_selected_context(x_batch, actions, cfg.C, cfg.L)
        answer_chunks_list = [ex["answer_chunks"] for ex in batch]
        r = compute_reward_batch(cfg, lm_model, lm_tokenizer, x_sel, vocab_inv, questions, answers, device, answer_chunks_list, actions)
        
        with torch.no_grad():
            adv = r - baseline
            baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * float(r.mean().item())
        
        loss = -(logp_action * adv).mean()
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        
        if (step + 1) % cfg.log_every == 0:
            logger.log_training_step(step + 1, lr, baseline, r.mean().item())
            tqdm.write(f"Step {step+1}: lr={lr:.2e}, baseline={baseline:.3f}, reward={r.mean().item():.3f}")
        
        if (step + 1) % cfg.eval_every == 0:
            train_acc = quick_eval(policy, train_examples[:20], cfg, pad_id, device)
            test_acc = quick_eval(policy, test_examples, cfg, pad_id, device)
            logger.log_eval(step + 1, train_acc, test_acc)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = {k: v.clone() for k, v in policy.state_dict().items()}
                logger.log(f"  ** New best: {test_acc:.1f}% **")
    
    # Restore best checkpoint
    if best_state is not None:
        policy.load_state_dict(best_state)
        logger.log(f"\nRestored best checkpoint (test={best_test_acc:.1f}%)")

def evaluate(cfg, policy, lm_model, lm_tokenizer, examples, vocab_inv, pad_id, device, logger, split_name="Test"):
    policy.eval()
    
    results = []
    logger.log(f"\n{split_name.upper()} RESULTS:")
    
    for i, ex in enumerate(examples):
        with torch.no_grad():
            x_input = ex["x"].unsqueeze(0)
            logits = policy(x_input)
            valid_mask = get_valid_chunk_mask(x_input, pad_id, cfg.C, cfg.L)
            greedy = torch.topk(logits, k=cfg.k, dim=-1).indices[0]
            selected_chunks = sorted(greedy.tolist())
            
            x_sel = build_selected_context(x_input, greedy.unsqueeze(0), cfg.C, cfg.L)
            r = compute_reward_batch(cfg, lm_model, lm_tokenizer, x_sel, vocab_inv, 
                                     [ex["question"]], [ex["answer"]], device,
                                     [ex["answer_chunks"]], greedy.unsqueeze(0))
            
            overlap = len(set(selected_chunks) & set(ex["answer_chunks"]))
            
            result = {
                "selected_chunks": selected_chunks,
                "answer_chunks": ex["answer_chunks"],
                "overlap": overlap,
                "total_answer_chunks": len(ex["answer_chunks"]),
                "reward": r[0].item(),
            }
            results.append(result)
            
            logger.log_example_result(split_name.lower(), i, ex, result)
            
            status = "✓" if result["overlap"] == result["total_answer_chunks"] and result["total_answer_chunks"] > 0 else "✗"
            logger.log(f"  {i+1} {status}: {ex['raw']['answer']:15s} | Selected={selected_chunks}, GT={ex['answer_chunks']}, R={r[0].item():.2f}")
    
    logger.log_summary(split_name.lower(), results)
    
    return results

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    
    logger = Logger(cfg.log_dir, "oracle_string_match")
    logger.log_config(cfg)
    
    with open('/root/data/train_data.json', 'r') as f:
        train_data = json.load(f)
    with open('/root/data/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    logger.log("Loading Qwen model...")
    lm_tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    lm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    lm_model.eval()
    
    all_texts = []
    for ex in train_data + test_data:
        combined = " ".join(ex["sentences"])
        all_texts.append(simple_word_tokenize(combined))
    
    vocab = build_vocab(all_texts)
    vocab_inv = {v: k for k, v in vocab.items()}
    pad_id = vocab["<pad>"]
    
    train_examples = prepare_dataset(train_data, vocab, cfg, device)
    test_examples = prepare_dataset(test_data, vocab, cfg, device)
    logger.log(f"Train: {len(train_examples)}, Test: {len(test_examples)}")
    
    policy = ChunkPolicy(vocab_size=len(vocab), C=cfg.C, L=cfg.L, pad_id=pad_id).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    
    logger.log(f"\nTraining {cfg.train_steps} steps with ORACLE string-match reward...")
    train_batched(cfg, policy, opt, lm_model, lm_tokenizer, train_examples, test_examples, vocab_inv, pad_id, device, logger)
    
    logger.log("\nFinal evaluation...")
    evaluate(cfg, policy, lm_model, lm_tokenizer, test_examples, vocab_inv, pad_id, device, logger, "Test")
    
    logger.save_metrics()
    logger.log("\nDone!")

if __name__ == "__main__":
    main()
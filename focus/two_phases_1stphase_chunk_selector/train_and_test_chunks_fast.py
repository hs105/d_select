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


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # chunking
    T: int = 96
    C: int = 12
    k: int = 2
    L: int = 8

    # training
    train_steps: int = 1000
    batch_size: int = 8
    lr: float = 1e-3
    baseline_momentum: float = 0.95

    model_name: str = "Qwen/Qwen2.5-3B"
    
    # logging
    log_dir: str = "/root/data/logs"
    log_every: int = 100


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
            "train_results": [],
            "test_results": [],
        }
        
        self.log(f"Experiment: {experiment_name}")
        self.log(f"Timestamp: {timestamp}")
        self.log("=" * 70)
    
    def log(self, message):
        """Write to both console and file"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_config(self, config):
        """Log configuration"""
        self.log("\nCONFIGURATION:")
        for key, value in vars(config).items():
            self.log(f"  {key}: {value}")
            self.metrics["config"][key] = str(value)
        self.log("")
    
    def log_training_step(self, step, baseline, avg_reward):
        """Log training progress"""
        self.metrics["training"].append({
            "step": step,
            "baseline": baseline,
            "avg_reward": avg_reward,
        })
    
    def log_example_result(self, split, idx, example, result):
        """Log detailed result for one example"""
        entry = {
            "split": split,
            "idx": idx,
            "question": example["raw"]["question"],
            "answer": example["raw"]["answer"],
            "selected_chunks": result["selected_chunks"],
            "answer_chunks": result["answer_chunks"],
            "overlap": result["overlap"],
            "total_answer_chunks": result["total_answer_chunks"],
            "reward": result["reward"],
            "status": "perfect" if result["overlap"] == result["total_answer_chunks"] and result["total_answer_chunks"] > 0 else ("partial" if result["overlap"] > 0 else "failed"),
        }
        
        if split == "train":
            self.metrics["train_results"].append(entry)
        else:
            self.metrics["test_results"].append(entry)
        
        return entry
    
    def log_summary(self, split, results):
        """Log summary statistics"""
        perfect = sum(1 for r in results if r["overlap"] == r["total_answer_chunks"] and r["total_answer_chunks"] > 0)
        partial = sum(1 for r in results if r["overlap"] > 0 and r["overlap"] < r["total_answer_chunks"])
        failed = sum(1 for r in results if r["overlap"] == 0)
        total = len(results)
        
        self.log(f"\n{split.upper()} SUMMARY:")
        self.log(f"  Perfect: {perfect}/{total} ({100*perfect/total:.1f}%)")
        self.log(f"  Partial: {partial}/{total} ({100*partial/total:.1f}%)")
        self.log(f"  Failed:  {failed}/{total} ({100*failed/total:.1f}%)")
        
        # Add to metrics
        summary_key = f"{split}_summary"
        self.metrics[summary_key] = {
            "perfect": perfect,
            "partial": partial,
            "failed": failed,
            "total": total,
            "perfect_pct": 100 * perfect / total if total > 0 else 0,
            "partial_pct": 100 * partial / total if total > 0 else 0,
            "failed_pct": 100 * failed / total if total > 0 else 0,
        }
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.log(f"\nMetrics saved to: {self.metrics_file}")
    
    def log_final_summary(self):
        """Log final summary"""
        self.log("\n" + "=" * 70)
        self.log("EXPERIMENT COMPLETE")
        self.log("=" * 70)
        self.log(f"Log file: {self.log_file}")
        self.log(f"Metrics file: {self.metrics_file}")


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
    batch_size = len(context_texts)
    logprobs = []
    
    mini_batch_size = 4
    for i in range(0, batch_size, mini_batch_size):
        batch_contexts = context_texts[i:i+mini_batch_size]
        batch_targets = target_tokens[i:i+mini_batch_size]
        
        inputs = lm_tokenizer(batch_contexts, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = lm_model(**inputs)
        
        seq_lens = inputs.attention_mask.sum(dim=1) - 1
        
        for j, (seq_len, target) in enumerate(zip(seq_lens, batch_targets)):
            logits = outputs.logits[j, seq_len, :]
            log_probs = F.log_softmax(logits, dim=-1)
            
            target_ids = lm_tokenizer.encode(target, add_special_tokens=False)
            if len(target_ids) == 0:
                target_ids = lm_tokenizer.encode(" " + target, add_special_tokens=False)
            if len(target_ids) == 0:
                logprobs.append(-100.0)
            else:
                logprobs.append(log_probs[target_ids[0]].item())
    
    return torch.tensor(logprobs, device=device)


def compute_reward_batch(cfg, lm_model, lm_tokenizer, x_selected_batch, vocab_inv, questions, answers, device):
    B = x_selected_batch.shape[0]
    context_texts = []
    target_tokens = []
    
    for i in range(B):
        selected_ids = x_selected_batch[i].cpu().tolist()
        selected_tokens = [vocab_inv.get(idx, "<unk>") for idx in selected_ids]
        selected_tokens = [t for t in selected_tokens if t != "<pad>"]
        
        if len(selected_tokens) == 0:
            context_text = questions[i]
        else:
            context_text = " ".join(selected_tokens) + " " + questions[i]
        
        context_texts.append(context_text)
        target_tokens.append(answers[i])
    
    return get_next_token_logprob_batch(lm_model, lm_tokenizer, context_texts, target_tokens, device)


class ChunkPolicy(nn.Module):
    def __init__(self, vocab_size: int, C: int, L: int, d_model: int = 64, pad_id: int = 0):
        super().__init__()
        self.C = C
        self.L = L
        self.pad_id = pad_id

        self.emb = nn.Embedding(vocab_size, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
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


def sample_k_without_replacement(logits: torch.Tensor, k: int):
    """
    FIXED: Sample k items without replacement
    Handles edge cases that cause NaN
    """
    B, C = logits.shape
    
    # Safety check
    if k > C:
        raise ValueError(f"Cannot sample k={k} items from C={C} chunks")
    
    actions, logps = [], []
    excluded = torch.zeros(B, C, dtype=torch.bool, device=logits.device)

    for step in range(k):
        # Mask already selected items
        masked_logits = logits.clone()
        masked_logits[excluded] = -1e9  # Use large negative instead of -inf
        
        # Check for NaN before softmax
        if torch.isnan(masked_logits).any():
            print(f"WARNING: NaN detected in logits at step {step}")
            print(f"Logits: {logits}")
            print(f"Excluded: {excluded}")
            # Fallback: sample uniformly from non-excluded
            available = ~excluded
            probs = available.float()
            probs = probs / probs.sum(dim=1, keepdim=True)
        else:
            probs = F.softmax(masked_logits, dim=-1)
        
        # Check for NaN after softmax
        if torch.isnan(probs).any() or (probs.sum(dim=1) == 0).any():
            print(f"WARNING: Invalid probs at step {step}")
            print(f"Probs: {probs}")
            # Fallback: uniform over available
            available = ~excluded
            probs = available.float()
            probs = probs / probs.sum(dim=1, keepdim=True)
        
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        actions.append(a)
        logps.append(dist.log_prob(a))
        
        # Mark as excluded
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


def train_batched(cfg, policy, opt, lm_model, lm_tokenizer, train_examples, vocab_inv, device, logger):
    import random
    
    baseline = -5.0
    
    for step in tqdm(range(cfg.train_steps), desc="Training"):
        batch = random.sample(train_examples, min(cfg.batch_size, len(train_examples)))
        
        x_batch = torch.stack([ex["x"] for ex in batch])
        questions = [ex["question"] for ex in batch]
        answers = [ex["answer"] for ex in batch]
        
        policy.train()
        
        logits = policy(x_batch)
        actions, logp_action = sample_k_without_replacement(logits, cfg.k)
        
        x_sel = build_selected_context(x_batch, actions, cfg.C, cfg.L)
        r = compute_reward_batch(cfg, lm_model, lm_tokenizer, x_sel, vocab_inv, questions, answers, device)
        
        with torch.no_grad():
            adv = r - baseline
            baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * float(r.mean().item())
        
        loss = -(logp_action * adv).mean()
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        
        if (step + 1) % cfg.log_every == 0:
            logger.log_training_step(step + 1, baseline, r.mean().item())
            tqdm.write(f"Step {step+1}: baseline={baseline:.3f}, avg_reward={r.mean().item():.3f}")


def evaluate(cfg, policy, lm_model, lm_tokenizer, examples, vocab_inv, device, logger, split_name="Test"):
    policy.eval()
    
    results = []
    logger.log(f"\n{'='*70}")
    logger.log(f"{split_name.upper()} RESULTS")
    logger.log(f"{'='*70}")
    
    for i, ex in enumerate(examples):
        with torch.no_grad():
            x_input = ex["x"].unsqueeze(0)
            logits = policy(x_input)
            greedy = torch.topk(logits, k=cfg.k, dim=-1).indices[0]
            selected_chunks = sorted(greedy.tolist())
            
            x_sel = build_selected_context(x_input, greedy.unsqueeze(0), cfg.C, cfg.L)
            r = compute_reward_batch(cfg, lm_model, lm_tokenizer, x_sel, vocab_inv, 
                                     [ex["question"]], [ex["answer"]], device)
            
            overlap = len(set(selected_chunks) & set(ex["answer_chunks"]))
            
            result = {
                "selected_chunks": selected_chunks,
                "answer_chunks": ex["answer_chunks"],
                "overlap": overlap,
                "total_answer_chunks": len(ex["answer_chunks"]),
                "reward": r[0].item(),
            }
            results.append(result)
            
            # Log to file
            logger.log_example_result(split_name.lower(), i, ex, result)
            
            # Print to console for test set or first few training examples
            if split_name == "Test" or (i < 5):
                status = "✓ PERFECT" if result["overlap"] == result["total_answer_chunks"] and result["total_answer_chunks"] > 0 else ("~ PARTIAL" if result["overlap"] > 0 else "✗ FAILED")
                logger.log(f"\n{split_name} {i+1}:")
                logger.log(f"  Question: {ex['raw']['question']}")
                logger.log(f"  Answer: {ex['raw']['answer']}")
                logger.log(f"  Selected: {selected_chunks}  Ground truth: {ex['answer_chunks']}")
                logger.log(f"  Overlap: {overlap}/{len(ex['answer_chunks'])}  Reward: {r[0].item():.3f}")
                logger.log(f"  {status}")
    
    logger.log_summary(split_name.lower(), results)
    
    return results


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    
    # Initialize logger
    logger = Logger(cfg.log_dir, "chunk_selection_experiment")
    logger.log_config(cfg)
    
    # Load data
    with open('/root/data/train_data.json', 'r') as f:
        train_data = json.load(f)
    with open('/root/data/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    logger.log("=" * 70)
    logger.log("LOADING QWEN MODEL")
    logger.log("=" * 70)
    lm_tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    lm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    lm_model.eval()
    logger.log("Model loaded!\n")
    
    # Build vocabulary
    logger.log("Building vocabulary...")
    all_texts = []
    for ex in train_data + test_data:
        combined = " ".join(ex["sentences"])
        all_texts.append(simple_word_tokenize(combined))
    
    vocab = build_vocab(all_texts)
    vocab_inv = {v: k for k, v in vocab.items()}
    logger.log(f"Vocabulary size: {len(vocab)}\n")
    
    # Prepare datasets
    logger.log("Preparing datasets...")
    train_examples = prepare_dataset(train_data, vocab, cfg, device)
    test_examples = prepare_dataset(test_data, vocab, cfg, device)
    logger.log(f"Train: {len(train_examples)}, Test: {len(test_examples)}\n")
    
    # Initialize policy
    policy = ChunkPolicy(vocab_size=len(vocab), C=cfg.C, L=cfg.L, pad_id=vocab["<pad>"]).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    
    # Training
    logger.log("=" * 70)
    logger.log(f"TRAINING ({cfg.train_steps} steps, batch_size={cfg.batch_size})")
    logger.log("=" * 70)
    train_batched(cfg, policy, opt, lm_model, lm_tokenizer, train_examples, vocab_inv, device, logger)
    
    # Evaluation
    logger.log("\nEvaluating on training set (first 10)...")
    evaluate(cfg, policy, lm_model, lm_tokenizer, train_examples[:10], vocab_inv, device, logger, "Train")
    
    logger.log("\nEvaluating on test set...")
    evaluate(cfg, policy, lm_model, lm_tokenizer, test_examples, vocab_inv, device, logger, "Test")
    
    # Save metrics and finish
    logger.save_metrics()
    logger.log_final_summary()


if __name__ == "__main__":
    main()

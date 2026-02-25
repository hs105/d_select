import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # chunking - 3 sentences with distractors
    T: int = 96   # Approximate length for 3 sentences
    C: int = 12   # 12 chunks (roughly 4 per sentence)
    k: int = 2    # Select 2 chunks
    L: int = 8    # Tokens per chunk

    # training
    train_steps: int = 500  # Steps per example
    batch_size: int = 1
    lr: float = 1e-3
    baseline_momentum: float = 0.95

    # Model will be set dynamically
    model_name: str = "Qwen/Qwen2.5-3B"


def simple_word_tokenize(text: str):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())


def build_vocab(all_tokens_list):
    """Build vocab from list of token lists"""
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


def get_next_token_logprob(lm_model, lm_tokenizer, context_text: str, target_token: str, device: str) -> float:
    with torch.no_grad():
        inputs = lm_tokenizer(context_text, return_tensors="pt").to(device)
        outputs = lm_model(**inputs)
        logits = outputs.logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        
        target_ids = lm_tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_ids) == 0:
            # Try with space prefix
            target_ids = lm_tokenizer.encode(" " + target_token, add_special_tokens=False)
        if len(target_ids) == 0:
            return -100.0
        
        return log_probs[target_ids[0]].item()


def compute_reward(cfg, lm_model, lm_tokenizer, x_selected, vocab_inv, question, answer, device):
    """Compute reward for selected chunks"""
    B = x_selected.shape[0]
    rewards = []
    
    for i in range(B):
        selected_ids = x_selected[i].cpu().tolist()
        selected_tokens = [vocab_inv.get(idx, "<unk>") for idx in selected_ids]
        selected_tokens = [t for t in selected_tokens if t != "<pad>"]
        
        if len(selected_tokens) == 0:
            context_text = question
        else:
            context_text = " ".join(selected_tokens) + " " + question
        
        log_prob = get_next_token_logprob(lm_model, lm_tokenizer, context_text, answer, device)
        rewards.append(log_prob)
    
    return torch.tensor(rewards, dtype=torch.float32, device=device)


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
    B, C = logits.shape
    actions, logps = [], []
    excluded = torch.zeros(B, C, dtype=torch.bool, device=logits.device)

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
    """Find which chunks contain the answer"""
    answer_chunks = []
    for chunk_idx in range(len(tokens) // L + 1):
        start = chunk_idx * L
        end = min(start + L, len(tokens))
        chunk_toks = tokens[start:end]
        if answer_lower in chunk_toks:
            answer_chunks.append(chunk_idx)
    return answer_chunks


def train_on_example(cfg, policy, opt, lm_model, lm_tokenizer, example, vocab, vocab_inv, device):
    """Train policy on a single example"""
    # Prepare input
    combined_text = " ".join(example["sentences"])
    tokens = simple_word_tokenize(combined_text)
    pad_id = vocab["<pad>"]
    
    x_input = prepare_sentence(tokens, vocab, cfg.T, pad_id).unsqueeze(0).to(device)
    
    # Get question and answer
    question = example["question"]
    answer = " " + example["answer"]  # Space prefix for tokenizer
    
    # Find ground truth answer chunks
    answer_chunks = find_answer_chunks(tokens, example["answer_lower"], cfg.L)
    
    # Training loop
    baseline = get_next_token_logprob(lm_model, lm_tokenizer, question, answer, device)
    
    for step in range(cfg.train_steps):
        policy.train()
        
        logits = policy(x_input)
        actions, logp_action = sample_k_without_replacement(logits, cfg.k)
        
        x_sel = build_selected_context(x_input, actions, cfg.C, cfg.L)
        r = compute_reward(cfg, lm_model, lm_tokenizer, x_sel, vocab_inv, question, answer, device)
        
        with torch.no_grad():
            adv = r - baseline
            baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * float(r.mean().item())
        
        loss = -(logp_action * adv).mean()
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    
    # Final evaluation
    with torch.no_grad():
        policy.eval()
        logits = policy(x_input)
        greedy = torch.topk(logits, k=cfg.k, dim=-1).indices[0]
        selected_chunks = sorted(greedy.tolist())
        
        x_sel_g = build_selected_context(x_input, greedy.unsqueeze(0), cfg.C, cfg.L)
        r_g = compute_reward(cfg, lm_model, lm_tokenizer, x_sel_g, vocab_inv, question, answer, device)
        
        overlap = len(set(selected_chunks) & set(answer_chunks))
        
        return {
            "selected_chunks": selected_chunks,
            "answer_chunks": answer_chunks,
            "overlap": overlap,
            "total_answer_chunks": len(answer_chunks),
            "reward": r_g[0].item(),
            "baseline": baseline,
        }


def test_on_example(cfg, policy, lm_model, lm_tokenizer, example, vocab, vocab_inv, device):
    """Test policy on a single example"""
    combined_text = " ".join(example["sentences"])
    tokens = simple_word_tokenize(combined_text)
    pad_id = vocab["<pad>"]
    
    x_input = prepare_sentence(tokens, vocab, cfg.T, pad_id).unsqueeze(0).to(device)
    
    question = example["question"]
    answer = " " + example["answer"]
    
    answer_chunks = find_answer_chunks(tokens, example["answer_lower"], cfg.L)
    
    with torch.no_grad():
        policy.eval()
        logits = policy(x_input)
        greedy = torch.topk(logits, k=cfg.k, dim=-1).indices[0]
        selected_chunks = sorted(greedy.tolist())
        
        x_sel = build_selected_context(x_input, greedy.unsqueeze(0), cfg.C, cfg.L)
        r = compute_reward(cfg, lm_model, lm_tokenizer, x_sel, vocab_inv, question, answer, device)
        
        overlap = len(set(selected_chunks) & set(answer_chunks))
        
        return {
            "selected_chunks": selected_chunks,
            "answer_chunks": answer_chunks,
            "overlap": overlap,
            "total_answer_chunks": len(answer_chunks),
            "reward": r[0].item(),
        }


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)
    device = cfg.device
    
    # Load data
    with open('/root/data/train_data.json', 'r') as f:
        train_data = json.load(f)
    with open('/root/data/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    print("=" * 70)
    print("LOADING QWEN MODEL")
    print("=" * 70)
    lm_tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    lm_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    lm_model.eval()
    print("Model loaded!\n")
    
    # Build vocabulary from all data
    print("Building vocabulary...")
    all_texts = []
    for ex in train_data + test_data:
        combined = " ".join(ex["sentences"])
        all_texts.append(simple_word_tokenize(combined))
    
    vocab = build_vocab(all_texts)
    vocab_inv = {v: k for k, v in vocab.items()}
    pad_id = vocab["<pad>"]
    print(f"Vocabulary size: {len(vocab)}\n")
    
    # Initialize policy
    policy = ChunkPolicy(vocab_size=len(vocab), C=cfg.C, L=cfg.L, pad_id=pad_id).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    
    # Training
    print("=" * 70)
    print(f"TRAINING ON {len(train_data)} EXAMPLES")
    print("=" * 70)
    
    train_results = []
    for i, example in enumerate(tqdm(train_data, desc="Training")):
        result = train_on_example(cfg, policy, opt, lm_model, lm_tokenizer, example, vocab, vocab_inv, device)
        train_results.append(result)
        
        if (i + 1) % 20 == 0:
            recent_20 = train_results[-20:]
            avg_overlap = sum(r["overlap"] for r in recent_20) / len(recent_20)
            avg_total = sum(r["total_answer_chunks"] for r in recent_20) / len(recent_20)
            print(f"\nExamples {i-19}-{i}: Avg overlap = {avg_overlap:.2f}/{avg_total:.2f}")
    
    # Training summary
    perfect = sum(1 for r in train_results if r["overlap"] == r["total_answer_chunks"])
    partial = sum(1 for r in train_results if r["overlap"] > 0 and r["overlap"] < r["total_answer_chunks"])
    failed = sum(1 for r in train_results if r["overlap"] == 0)
    
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Perfect (found all answer chunks): {perfect}/{len(train_results)} ({100*perfect/len(train_results):.1f}%)")
    print(f"Partial (found some answer chunks): {partial}/{len(train_results)} ({100*partial/len(train_results):.1f}%)")
    print(f"Failed (found no answer chunks):   {failed}/{len(train_results)} ({100*failed/len(train_results):.1f}%)")
    
    # Testing
    print("\n" + "=" * 70)
    print(f"TESTING ON {len(test_data)} EXAMPLES")
    print("=" * 70)
    
    test_results = []
    for i, example in enumerate(test_data):
        result = test_on_example(cfg, policy, lm_model, lm_tokenizer, example, vocab, vocab_inv, device)
        test_results.append(result)
        
        print(f"\nTest {i+1}:")
        print(f"  Question: {example['question']}")
        print(f"  Answer: {example['answer']}")
        print(f"  Selected chunks: {result['selected_chunks']}")
        print(f"  Answer chunks: {result['answer_chunks']}")
        print(f"  Overlap: {result['overlap']}/{result['total_answer_chunks']}")
        print(f"  Reward: {result['reward']:.3f}")
        
        if result['overlap'] == result['total_answer_chunks']:
            print("  ✓ SUCCESS")
        elif result['overlap'] > 0:
            print("  ~ PARTIAL")
        else:
            print("  ✗ FAILED")
    
    # Test summary
    perfect = sum(1 for r in test_results if r["overlap"] == r["total_answer_chunks"])
    partial = sum(1 for r in test_results if r["overlap"] > 0 and r["overlap"] < r["total_answer_chunks"])
    failed = sum(1 for r in test_results if r["overlap"] == 0)
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Perfect: {perfect}/{len(test_results)} ({100*perfect/len(test_results):.1f}%)")
    print(f"Partial: {partial}/{len(test_results)} ({100*partial/len(test_results):.1f}%)")
    print(f"Failed:  {failed}/{len(test_results)} ({100*failed/len(test_results):.1f}%)")


if __name__ == "__main__":
    main()

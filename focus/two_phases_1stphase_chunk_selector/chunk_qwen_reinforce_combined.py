import re
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
REINFORCE Algorithm for Chunk Selection with Language Model Feedback

This script demonstrates using REINFORCE (policy gradient) to learn which chunks
of text to select to maximize a language model's prediction accuracy.

SETUP:
------
- Input: Three sentences concatenated into one sequence (96 tokens, 12 chunks)
- Task: Select k=2 chunks that help Qwen predict "Paris" when asked 
        "The capital city of France is"
- Policy: Small neural network that scores each chunk, samples k without replacement
- Reward: log P(" Paris" | selected_chunks + query) from Qwen-2.5-3B

ALGORITHM (REINFORCE with Baseline):
------------------------------------
At each training step:
1. Policy network scores all chunks: logits = policy(x)  
2. Sample k chunks without replacement: actions ~ Categorical(softmax(logits))
3. Get log probability of the sampled action: log π_θ(actions)
4. Compute reward: r = log P_LM(" Paris" | selected_chunks + query)
5. Compute advantage: adv = r - baseline
   - baseline = exponential moving average of past rewards
   - Reduces variance: advantages are relative to typical performance
6. Update policy: ∇_θ J ≈ log π_θ(actions) * adv
   - Good actions (r > baseline) → increase probability
   - Bad actions (r < baseline) → decrease probability
7. Update baseline: baseline ← 0.95 * baseline + 0.05 * r

KEY INSIGHT - Why Baseline Matters:
-----------------------------------
Without baseline:
  All rewards are negative (log probs: -0.1, -5.0, -10.0)
  → Gradient always decreases all action probabilities
  → Algorithm fails to distinguish good from bad actions

With baseline (e.g., baseline = -2.0):
  r = -0.1  → adv = -0.1 - (-2.0) = +1.9  → INCREASE this action's probability
  r = -5.0  → adv = -5.0 - (-2.0) = -3.0  → DECREASE this action's probability
  → Algorithm learns to prefer better actions

EXPECTED BEHAVIOR:
-----------------
- Step 1-100: Baseline adjusts from initial value to true average reward
- Step 100-400: Policy explores and converges to optimal chunks
- Step 400+: Stable convergence to chunks [1, 2] from Sentence 1
  - Chunk 1: ['france', 'is', 'paris', ...] 
  - Chunk 2: ['paris', 'every', 'year', ...]
  - Reward: -0.166 (ranks #2 out of 66 possible 2-chunk combinations)

The policy successfully learns to:
1. Select chunks from Sentence 1 (not Sentences 2-3)
2. Select chunks containing "Paris"
3. Ignore chunks that confuse the language model

RESULTS:
--------
Final selection: Chunks [1, 2] with reward -0.166
- From Sentence 1: [1, 2] ✓
- From Sentences 2-3: [] ✓
- Contains "Paris": Yes ✓
- Near-optimal: Rank #2/66 ✓


Note:
Baseline = bias you subtract to get advantage. 


Author: Hengshuai Yao
Date: 2026-02-07
"""

@dataclass
class Config:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # chunking - NOW MUCH LARGER to fit all 3 sentences
    T: int = 96  # 3 sentences * 32 tokens = 96 tokens total
    C: int = 12  # 12 chunks total (4 per sentence)
    k: int = 2   # Still select 2 chunks
    L: int = 8   # Tokens per chunk

    # training
    steps: int = 2000
    batch_size: int = 1  # Just one combined input
    lr: float = 1e-3
    baseline_momentum: float = 0.95
    log_every: int = 100

    # Query
    query: str = "The capital city of France is"
    target_answer: str = " Paris"

    keyword: str = "paris"


def simple_word_tokenize(text: str):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())


def build_vocab(all_tokens):
    vocab = {"<pad>": 0, "<unk>": 1}
    for t in all_tokens:
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
            return -100.0
        
        return log_probs[target_ids[0]].item()


def compute_reward_with_lm(cfg: Config, lm_model, lm_tokenizer, 
                            x_selected: torch.Tensor, vocab_inv: dict, device: str) -> torch.Tensor:
    B = x_selected.shape[0]
    rewards = []
    
    for i in range(B):
        selected_ids = x_selected[i].cpu().tolist()
        selected_tokens = [vocab_inv.get(idx, "<unk>") for idx in selected_ids]
        selected_tokens = [t for t in selected_tokens if t != "<pad>"]
        
        if len(selected_tokens) == 0:
            context_text = cfg.query
        else:
            context_text = " ".join(selected_tokens) + " " + cfg.query
        
        log_prob = get_next_token_logprob(lm_model, lm_tokenizer, context_text, cfg.target_answer, device)
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


def analyze_chunks(tokens, keyword, L, C):
    """Analyze which chunks contain the keyword"""
    kw_positions = [i for i, t in enumerate(tokens) if t == keyword]
    kw_chunks = sorted(set([pos // L for pos in kw_positions]))
    
    # Show which sentence each chunk belongs to
    chunk_to_sentence = []
    for c in range(C):
        # Roughly: chunks 0-3 = sentence 1, 4-7 = sentence 2, 8-11 = sentence 3
        if c < 4:
            chunk_to_sentence.append(1)
        elif c < 8:
            chunk_to_sentence.append(2)
        else:
            chunk_to_sentence.append(3)
    
    return kw_positions, kw_chunks, chunk_to_sentence


def main():
    # Three sentences - will be concatenated into ONE input
    sentences = [
        "In a trivia quiz, the capital of France is Paris, and many tourists visit Paris every year.",
        "Many museums and galleries throughout Paris display incredible art, while popular cafes in Paris serve delicious pastries.",
        "Paris is known for its romantic atmosphere, and lovers often stroll through Paris at sunset.",
    ]
    
    cfg = Config()
    assert cfg.T == cfg.C * cfg.L

    torch.manual_seed(cfg.seed)
    device = cfg.device

    print("=" * 70)
    print("LOADING QWEN MODEL")
    print("=" * 70)
    
    model_name = "Qwen/Qwen2.5-3B"
    print(f"Loading {model_name}...")
    lm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    lm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    lm_model.eval()
    print("Model loaded!\n")

    # Concatenate all sentences into one long text
    combined_text = " ".join(sentences)
    all_tokens = simple_word_tokenize(combined_text)
    
    # Build vocabulary
    vocab = build_vocab(all_tokens)
    vocab_inv = {v: k for k, v in vocab.items()}
    pad_id = vocab["<pad>"]

    print("=" * 70)
    print("COMBINED INPUT ANALYSIS")
    print("=" * 70)
    print(f"Total tokens: {len(all_tokens)}")
    print(f"Total chunks: {cfg.C} (with L={cfg.L} tokens per chunk)")
    print(f"Selecting k={cfg.k} chunks\n")
    
    kw_positions, kw_chunks, chunk_to_sentence = analyze_chunks(all_tokens, cfg.keyword, cfg.L, cfg.C)
    print(f"Keyword '{cfg.keyword}' appears at positions: {kw_positions}")
    print(f"Keyword appears in chunks: {kw_chunks}\n")
    
    print("Chunk breakdown:")
    for c in range(cfg.C):
        start = c * cfg.L
        end = min(start + cfg.L, len(all_tokens))
        chunk_toks = all_tokens[start:end]
        has_kw = cfg.keyword in chunk_toks
        sent_num = chunk_to_sentence[c]
        marker = f" ← HAS '{cfg.keyword.upper()}'" if has_kw else ""
        print(f"  Chunk {c:2d} (Sentence {sent_num}): {chunk_toks}{marker}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)
    print(f"Chunks from Sentence 1 (chunks 0-3): Should contain answer")
    print(f"  Chunks with 'paris': {[c for c in kw_chunks if c < 4]}")
    print(f"Chunks from Sentences 2-3 (chunks 4-11): May confuse the model")
    print(f"  Chunks with 'paris': {[c for c in kw_chunks if c >= 4]}")
    
    # Test baseline
    print("\n" + "=" * 70)
    print("BASELINE TEST")
    print("=" * 70)
    baseline_logprob = get_next_token_logprob(lm_model, lm_tokenizer, cfg.query, cfg.target_answer, device)
    print(f"Query only: '{cfg.query}'")
    print(f"Log P(' Paris'): {baseline_logprob:.4f}\n")

    # Prepare the combined input
    x_combined = prepare_sentence(all_tokens, vocab, cfg.T, pad_id).unsqueeze(0).to(device)  # [1, T]

    baseline = baseline_logprob
    policy = ChunkPolicy(vocab_size=len(vocab), C=cfg.C, L=cfg.L, pad_id=pad_id).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for step in range(1, cfg.steps + 1):
        policy.train()
        
        logits = policy(x_combined)  # [1, C]
        actions, logp_action = sample_k_without_replacement(logits, cfg.k)

        x_sel = build_selected_context(x_combined, actions, cfg.C, cfg.L)
        r = compute_reward_with_lm(cfg, lm_model, lm_tokenizer, x_sel, vocab_inv, device)

        with torch.no_grad():
            adv = r - baseline
            baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * float(r.mean().item())

        loss = -(logp_action * adv).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == 1:
            with torch.no_grad():
                greedy = torch.topk(logits, k=cfg.k, dim=-1).indices[0]  # [k]
                x_sel_g = build_selected_context(x_combined, greedy.unsqueeze(0), cfg.C, cfg.L)
                r_g = compute_reward_with_lm(cfg, lm_model, lm_tokenizer, x_sel_g, vocab_inv, device)
                
                selected_chunks = sorted(greedy.tolist())
                sampled_chunks = sorted(actions[0].tolist())
                
                # Analyze which sentence the chunks come from
                sent1_chunks = [c for c in selected_chunks if c < 4]
                sent23_chunks = [c for c in selected_chunks if c >= 4]
                
                overlap_with_keyword = len(set(selected_chunks) & set(kw_chunks))
                overlap_with_sent1_kw = len(set(sent1_chunks) & set([c for c in kw_chunks if c < 4]))
                
                print(f"\nStep {step}:")
                print(f"  Sampled chunks: {sampled_chunks}  r={r[0].item():.3f}")
                print(f"  Greedy chunks:  {selected_chunks}  r={r_g[0].item():.3f}")
                print(f"  From Sent1: {sent1_chunks}  From Sent2-3: {sent23_chunks}")
                print(f"  Overlap with keyword chunks: {overlap_with_keyword}/{len(kw_chunks)}")
                print(f"  Overlap with Sent1 keyword chunks: {overlap_with_sent1_kw}/{len([c for c in kw_chunks if c < 4])}")
                print(f"  Baseline: {baseline:.3f}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    with torch.no_grad():
        policy.eval()
        logits = policy(x_combined)
        greedy = torch.topk(logits, k=cfg.k, dim=-1).indices[0]
        
        selected_chunks = sorted(greedy.tolist())
        sent1_chunks = [c for c in selected_chunks if c < 4]
        sent1_keyword_chunks = [c for c in kw_chunks if c < 4]
        
        print(f"\nSelected chunks: {selected_chunks}")
        print(f"  From Sentence 1 (chunks 0-3): {sent1_chunks}")
        print(f"  From Sentences 2-3 (chunks 4-11): {[c for c in selected_chunks if c >= 4]}")
        print(f"\nKeyword chunks in Sentence 1: {sent1_keyword_chunks}")
        
        overlap = len(set(sent1_chunks) & set(sent1_keyword_chunks))
        print(f"Overlap with Sent1 keyword chunks: {overlap}/{len(sent1_keyword_chunks)}")
        
        if overlap == len(sent1_keyword_chunks):
            print("\n✓ SUCCESS! Policy learned to select chunks from Sentence 1 containing 'Paris'")
        else:
            print(f"\n✗ Partial success: Found {overlap}/{len(sent1_keyword_chunks)} correct chunks")
        
        # Show what was selected
        print("\nSelected context:")
        for chunk_idx in selected_chunks:
            start = chunk_idx * cfg.L
            end = min(start + cfg.L, len(all_tokens))
            chunk_toks = all_tokens[start:end]
            has_kw = cfg.keyword in chunk_toks
            marker = f" ← HAS '{cfg.keyword.upper()}'" if has_kw else ""
            print(f"  Chunk {chunk_idx}: {chunk_toks}{marker}")


if __name__ == "__main__":
    main()

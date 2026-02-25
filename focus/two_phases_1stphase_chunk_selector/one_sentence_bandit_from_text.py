import re
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # chunking - reduced to fit shorter sentences
    T: int = 32  # Total sequence length
    C: int = 4   # Number of chunks
    k: int = 2   # Select 2 chunks
    L: int = 8   # Tokens per chunk (must satisfy T == C * L)

    # training
    steps: int = 1000
    batch_size: int = 2  # Process both sentences each step
    lr: float = 1e-3
    baseline_momentum: float = 0.95
    log_every: int = 100

    # dummy frozen "LM" oracle:
    keyword: str = "paris"
    p_hit: float = 0.6
    p_miss: float = 0.02

    keyword_chunk_count: int = 2


def simple_word_tokenize(text: str):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())


def build_vocab(all_tokens):
    vocab = {"<pad>": 0, "<unk>": 1}
    for tokens in all_tokens:
        for t in tokens:
            if t not in vocab:
                vocab[t] = len(vocab)
    return vocab


def encode(tokens, vocab):
    return [vocab.get(t, vocab["<unk>"]) for t in tokens]


def prepare_sentence(tokens, vocab, T, pad_id):
    """Convert tokens to fixed-length tensor"""
    ids = encode(tokens, vocab)
    if len(ids) < T:
        ids = ids + [pad_id] * (T - len(ids))
    else:
        ids = ids[:T]
    return torch.tensor(ids, dtype=torch.long)


def dummy_logp_true(cfg: Config, x_selected: torch.Tensor, keyword_id: int) -> torch.Tensor:
    hit_count = (x_selected == keyword_id).sum(dim=1).float()  # [B], count of keyword occurrences
    
    # Option 1: Reward proportional to number of hits
    # More "paris" = higher reward
    base_reward = torch.log(torch.tensor(cfg.p_hit, device=x_selected.device))
    miss_penalty = torch.log(torch.tensor(cfg.p_miss, device=x_selected.device))
    
    # Linear reward: reward increases with each additional hit
    reward = torch.where(
        hit_count > 0,
        base_reward + 0.1 * hit_count,  # Bonus for more hits
        miss_penalty
    )
    
    return reward


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
        """
        x: [B,T] -> logits [B,C]
        """
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


def comb(n, r):
    from math import comb as _comb
    return _comb(n, r)


def expected_reward_random(cfg: Config) -> float:
    C, k, m = cfg.C, cfg.k, cfg.keyword_chunk_count
    p_hit = 1.0 - (comb(C - m, k) / comb(C, k))
    return p_hit * math.log(cfg.p_hit) + (1.0 - p_hit) * math.log(cfg.p_miss)


def analyze_sentence(tokens, keyword, L):
    """Show where keyword appears in chunks"""
    kw_positions = [i for i, t in enumerate(tokens) if t == keyword]
    kw_chunks = sorted(set([pos // L for pos in kw_positions]))
    return kw_positions, kw_chunks


def main():
    # Two sentences to test generalization
    sentences = [
        "In a trivia quiz, the capital of France is Paris, and many tourists visit Paris every year.",
        "Many museums and galleries throughout Paris display incredible art, while popular cafes in Paris serve delicious pastries.",
        "Paris is known for its romantic atmosphere, and lovers often stroll through Paris at sunset.",
    ]
    
    cfg = Config()
    assert cfg.T == cfg.C * cfg.L

    torch.manual_seed(cfg.seed)
    device = cfg.device

    # Tokenize all sentences
    all_tokens = [simple_word_tokenize(s) for s in sentences]
    
    # Build vocabulary from all sentences
    vocab = build_vocab(all_tokens)
    pad_id = vocab["<pad>"]
    keyword_id = vocab[cfg.keyword]

    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    for i, (sent, tokens) in enumerate(zip(sentences, all_tokens)):
        print(f"\nSentence {i+1} ({len(tokens)} tokens):")
        print(f"Text: {sent}")
        kw_pos, kw_chunks = analyze_sentence(tokens, cfg.keyword, cfg.L)
        print(f"Keyword '{cfg.keyword}' at positions {kw_pos} -> chunks {kw_chunks}")
        print(f"Chunks (L={cfg.L}):")
        for c in range(cfg.C):
            start = c * cfg.L
            end = min(start + cfg.L, len(tokens))
            chunk_toks = tokens[start:end]
            has_kw = cfg.keyword in chunk_toks
            marker = " ← HAS KEYWORD" if has_kw else ""
            print(f"  Chunk {c}: {chunk_toks}{marker}")
    print("\n" + "=" * 70)

    # Prepare all sentences as tensors
    sentence_tensors = [prepare_sentence(tokens, vocab, cfg.T, pad_id).to(device) 
                        for tokens in all_tokens]
    
    # Stack into batch [2, T]
    x_batch = torch.stack(sentence_tensors)

    baseline = expected_reward_random(cfg)
    print(f"Baseline initialized to: {baseline:.4f}\n")

    policy = ChunkPolicy(vocab_size=len(vocab), C=cfg.C, L=cfg.L, pad_id=pad_id).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for step in range(1, cfg.steps + 1):
        policy.train()
        
        # Process both sentences in a batch
        logits = policy(x_batch)  # [2, C]
        actions, logp_action = sample_k_without_replacement(logits, cfg.k)

        x_sel = build_selected_context(x_batch, actions, cfg.C, cfg.L)
        r = dummy_logp_true(cfg, x_sel, keyword_id=keyword_id)  # [2]

        with torch.no_grad():
            adv = r - baseline
            baseline = cfg.baseline_momentum * baseline + (1 - cfg.baseline_momentum) * float(r.mean().item())

        loss = -(logp_action * adv).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.log_every == 0 or step == 1:
            with torch.no_grad():
                greedy = torch.topk(logits, k=cfg.k, dim=-1).indices
                x_sel_g = build_selected_context(x_batch, greedy, cfg.C, cfg.L)
                r_g = dummy_logp_true(cfg, x_sel_g, keyword_id=keyword_id)
                
                print(f"\nStep {step}:")
                print(f"  Sentence 1: sampled={actions[0].tolist()}  greedy={greedy[0].tolist()}  "
                      f"r_sample={r[0].item():.3f}  r_greedy={r_g[0].item():.3f}")
                print(f"  Sentence 2: sampled={actions[1].tolist()}  greedy={greedy[1].tolist()}  "
                      f"r_sample={r[1].item():.3f}  r_greedy={r_g[1].item():.3f}")
                print(f"  Baseline: {baseline:.3f}")

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    with torch.no_grad():
        policy.eval()
        logits = policy(x_batch)
        greedy = torch.topk(logits, k=cfg.k, dim=-1).indices
        
        for i in range(len(sentences)):
            kw_pos, kw_chunks = analyze_sentence(all_tokens[i], cfg.keyword, cfg.L)
            print(f"\nSentence {i+1}:")
            print(f"  True keyword chunks: {kw_chunks}")
            print(f"  Policy selected chunks: {sorted(greedy[i].tolist())}")
            selected_chunks = set(greedy[i].tolist())
            true_chunks = set(kw_chunks)
            overlap = selected_chunks & true_chunks
            print(f"  Overlap: {len(overlap)}/{len(true_chunks)} keyword chunks found")


if __name__ == "__main__":
    main()
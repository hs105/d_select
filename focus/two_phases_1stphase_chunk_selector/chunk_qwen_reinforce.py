import re
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # chunking
    T: int = 32
    C: int = 4
    k: int = 2
    L: int = 8

    # training
    steps: int = 1000
    batch_size: int = 3
    lr: float = 1e-3
    baseline_momentum: float = 0.95
    log_every: int = 100

    # Query for the LM - FIXED: Use capitalized Paris
    query: str = "The capital city of France is"
    target_answer: str = " Paris"  # Note the space! This matches tokenization

    keyword: str = "paris"
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
        
        # Tokenize target - this handles the case sensitivity correctly
        target_ids = lm_tokenizer.encode(target_token, add_special_tokens=False)
        
        if len(target_ids) == 0:
            print(f"Warning: Could not tokenize target '{target_token}'")
            return -100.0
        
        target_id = target_ids[0]
        return log_probs[target_id].item()


def compute_reward_with_lm(cfg: Config, lm_model, lm_tokenizer, 
                            x_selected: torch.Tensor, vocab_inv: dict, device: str) -> torch.Tensor:
    B = x_selected.shape[0]
    rewards = []
    
    for i in range(B):
        selected_ids = x_selected[i].cpu().tolist()
        selected_tokens = [vocab_inv.get(idx, "<unk>") for idx in selected_ids]
        selected_tokens = [t for t in selected_tokens if t != "<pad>"]
        
        # Create context
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


def analyze_sentence(tokens, keyword, L):
    kw_positions = [i for i, t in enumerate(tokens) if t == keyword]
    kw_chunks = sorted(set([pos // L for pos in kw_positions]))
    return kw_positions, kw_chunks


def main():
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

    all_tokens = [simple_word_tokenize(s) for s in sentences]
    vocab = build_vocab(all_tokens)
    vocab_inv = {v: k for k, v in vocab.items()}
    pad_id = vocab["<pad>"]

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

    print("\nTesting LM with query only:")
    baseline_logprob = get_next_token_logprob(lm_model, lm_tokenizer, cfg.query, cfg.target_answer, device)
    print(f"Query: '{cfg.query}'")
    print(f"Target: '{cfg.target_answer}'")
    print(f"Log P(target): {baseline_logprob:.4f}\n")

    sentence_tensors = [prepare_sentence(tokens, vocab, cfg.T, pad_id).to(device) 
                        for tokens in all_tokens]
    x_batch = torch.stack(sentence_tensors)

    baseline = baseline_logprob
    print(f"Baseline initialized to: {baseline:.4f}\n")

    policy = ChunkPolicy(vocab_size=len(vocab), C=cfg.C, L=cfg.L, pad_id=pad_id).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for step in range(1, cfg.steps + 1):
        policy.train()
        
        logits = policy(x_batch)
        actions, logp_action = sample_k_without_replacement(logits, cfg.k)

        x_sel = build_selected_context(x_batch, actions, cfg.C, cfg.L)
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
                greedy = torch.topk(logits, k=cfg.k, dim=-1).indices
                x_sel_g = build_selected_context(x_batch, greedy, cfg.C, cfg.L)
                r_g = compute_reward_with_lm(cfg, lm_model, lm_tokenizer, x_sel_g, vocab_inv, device)
                
                print(f"\nStep {step}:")
                for i in range(len(sentences)):
                    print(f"  Sentence {i+1}: sampled={actions[i].tolist()}  greedy={greedy[i].tolist()}  "
                          f"r_sample={r[i].item():.3f}  r_greedy={r_g[i].item():.3f}")
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
            
            selected_ids = build_selected_context(x_batch[i:i+1], greedy[i:i+1], cfg.C, cfg.L)[0].cpu().tolist()
            selected_tokens = [vocab_inv.get(idx, "<unk>") for idx in selected_ids if vocab_inv.get(idx) != "<pad>"]
            print(f"  Selected context: {' '.join(selected_tokens)}")


if __name__ == "__main__":
    main()

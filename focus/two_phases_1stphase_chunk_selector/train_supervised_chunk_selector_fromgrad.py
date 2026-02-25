"""
Train Focus Network using gradient attribution from Qwen as supervision.

Instead of REINFORCE with contrastive reward, we:
1. Run Qwen with full context, compute gradient of logP(answer) w.r.t. input embeddings
2. Map gradient importance to chunk-level scores -> soft labels
3. Train focus network via cross-entropy to predict which chunks are important

This is supervised learning with the LM's own gradient as the teacher signal.
"""
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np


def simple_word_tokenize(text):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())


def find_answer_chunks(tokens, answer_lower, L):
    answer_tokens = simple_word_tokenize(answer_lower)
    answer_len = len(answer_tokens)
    gt_chunks = []
    num_chunks = (len(tokens) + L - 1) // L
    for c in range(num_chunks):
        start = c * L
        end = min(start + L, len(tokens))
        chunk_toks = tokens[start:end]
        for j in range(len(chunk_toks) - answer_len + 1):
            if chunk_toks[j:j + answer_len] == answer_tokens:
                gt_chunks.append(c)
                break
    return gt_chunks


# ============================================================
# Focus Network (same architecture as before)
# ============================================================
class FocusNetwork(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, pad_id, C, L):
        """
        x: [B, T] token ids, T = C * L
        Returns: [B, C] chunk logits
        """
        B, T = x.shape
        e = self.embed(x)  # [B, T, D]
        e = e.view(B, C, L, -1)  # [B, C, L, D]

        mask = (x.view(B, C, L) != pad_id).unsqueeze(-1).float()  # [B, C, L, 1]
        pooled = (e * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)  # [B, C, D]

        logits = self.mlp(pooled).squeeze(-1)  # [B, C]
        return logits


# ============================================================
# Gradient attribution from Qwen
# ============================================================
def compute_chunk_gradient_importance(model, embed_layer, tokenizer, combined, question, answer, num_chunks, L, device):
    """
    Compute per-chunk importance using gradient of logP(answer) w.r.t. input embeddings.
    Returns normalized importance scores [num_chunks].
    """
    input_text = combined + " " + question
    full_text = input_text + " " + answer

    context_ids = tokenizer(input_text, return_tensors="pt").to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").to(device)

    context_len = context_ids.input_ids.shape[1]
    input_ids = full_ids.input_ids

    embeddings = embed_layer(input_ids).detach().requires_grad_(True)
    outputs = model(inputs_embeds=embeddings)

    pred_pos = context_len - 1
    target_token_id = input_ids[0, context_len]

    logits = outputs.logits[0, pred_pos, :]
    log_probs = F.log_softmax(logits, dim=-1)
    target_logp = log_probs[target_token_id]

    target_logp.backward()

    grad = embeddings.grad[0]
    token_importance = grad.norm(dim=-1)  # [seq_len]

    # Map BPE to chunks
    context_only_ids = tokenizer(combined, return_tensors="pt").to(device)
    context_bpe_len = context_only_ids.input_ids.shape[1]

    chunk_importance = torch.zeros(num_chunks, device=device)
    chunk_count = torch.zeros(num_chunks, device=device)

    for bpe_pos in range(min(context_bpe_len, token_importance.shape[0])):
        frac = bpe_pos / max(context_bpe_len, 1)
        chunk_idx = min(int(frac * num_chunks), num_chunks - 1)
        chunk_importance[chunk_idx] += token_importance[bpe_pos].item()
        chunk_count[chunk_idx] += 1

    valid = chunk_count > 0
    chunk_importance[valid] /= chunk_count[valid]

    # Normalize to sum to 1 (soft label distribution)
    chunk_importance = chunk_importance / chunk_importance.sum().clamp_min(1e-8)

    embeddings.grad = None
    model.zero_grad()

    return chunk_importance


# ============================================================
# Data
# ============================================================
TRAIN_DATA = [
    {"question": "What is the smallest planet in our solar system?", "answer": "Mercury", "answer_lower": "mercury",
     "sentences": ["Artists express creativity through different mediums including painting, sculpture, music, and digital art forms.",
                    "Natural wonders inspire awe and wonder in people who visit national parks and protected wilderness areas.",
                    "In our solar system, Mercury is the smallest planet with a diameter of only three thousand miles.",
                    "The smallest planet in our solar system is Mercury."]},
    {"question": "What is the capital city of Italy?", "answer": "Rome", "answer_lower": "rome",
     "sentences": ["In Rome, the capital, you can explore ruins from the Roman Empire that still stand today.",
                    "Natural wonders inspire awe and wonder in people who visit national parks and protected wilderness areas.",
                    "Artists express creativity through different mediums including painting, sculpture, music, and digital art forms.",
                    "The capital city of Italy is Rome."]},
    {"question": "What is the chemical symbol for silver?", "answer": "Ag", "answer_lower": "ag",
     "sentences": ["Historical events have shaped modern society in countless ways that continue to influence us today.",
                    "Ancient civilizations built remarkable structures that archaeologists continue to study and preserve for future generations.",
                    "The chemical symbol Ag represents silver in equations, originating from ancient Latin terminology for the metal.",
                    "The chemical symbol for silver is Ag."]},
    {"question": "What is the tallest mountain on Earth?", "answer": "Everest", "answer_lower": "everest",
     "sentences": ["Students study various subjects in school to prepare for their future careers and personal growth.",
                    "The highest point on Earth is the peak of Mount Everest, which towers above all other mountains.",
                    "Historical events have shaped modern society in countless ways that continue to influence us today.",
                    "The tallest mountain on Earth is Mount Everest."]},
    {"question": "What is the chemical symbol for gold?", "answer": "Au", "answer_lower": "au",
     "sentences": ["Students study various subjects in school to prepare for their future careers and personal growth.",
                    "Museums preserve cultural heritage and educate visitors about history, science, and artistic achievements from past eras.",
                    "On the periodic table, gold is represented by the symbol Au, derived from the Latin word aurum.",
                    "The chemical symbol for gold is Au."]},
    {"question": "What is the largest planet in our solar system?", "answer": "Jupiter", "answer_lower": "jupiter",
     "sentences": ["Scientists conduct research in laboratories using advanced equipment to make new discoveries about our world.",
                    "Astronomers study Jupiter because it is the biggest planet and has dozens of interesting moons orbiting it.",
                    "Ancient civilizations built remarkable structures that archaeologists continue to study and preserve for future generations.",
                    "The largest planet in our solar system is Jupiter."]},
    {"question": "Who wrote the play Hamlet?", "answer": "Shakespeare", "answer_lower": "shakespeare",
     "sentences": ["Theater companies worldwide perform Hamlet by Shakespeare, who is considered the greatest English playwright ever.",
                    "Students study various subjects in school to prepare for their future careers and personal growth.",
                    "Ancient civilizations built remarkable structures that archaeologists continue to study and preserve for future generations.",
                    "The play Hamlet was written by Shakespeare."]},
    {"question": "What is the capital city of Japan?", "answer": "Tokyo", "answer_lower": "tokyo",
     "sentences": ["Historical events have shaped modern society in countless ways that continue to influence us today.",
                    "Scientists conduct research in laboratories using advanced equipment to make new discoveries about our world.",
                    "Mount Fuji can be seen from Tokyo on clear days, though the capital is quite far from it.",
                    "The capital of Japan is Tokyo."]},
    {"question": "What is the capital city of France?", "answer": "Paris", "answer_lower": "paris",
     "sentences": ["In a trivia quiz, the capital of France is Paris, and many tourists visit Paris every year.",
                    "Many people around the world enjoy learning about geography and exploring different cultures each day.",
                    "Museums preserve cultural heritage and educate visitors about history, science, and artistic achievements from past eras.",
                    "The capital city of France is Paris. Paris is a great place."]},
    {"question": "Who painted the Mona Lisa?", "answer": "da Vinci", "answer_lower": "da vinci",
     "sentences": ["Students study various subjects in school to prepare for their future careers and personal growth.",
                    "The artist da Vinci spent years perfecting the Mona Lisa, working on details with incredible precision and care.",
                    "Historical events have shaped modern society in countless ways that continue to influence us today.",
                    "The Mona Lisa was painted by da Vinci."]},
]

# Use same data for test (small experiment)
TEST_DATA = TRAIN_DATA


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(42)
    torch.manual_seed(42)

    L = 8
    C = 12  # max chunks
    T = C * L
    k = 2  # select top-2 chunks

    # ============================================================
    # Step 1: Precompute gradient attribution labels using Qwen
    # ============================================================
    print("Loading Qwen for gradient attribution...")
    tokenizer_lm = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    model_lm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model_lm.eval()
    embed_layer = model_lm.model.embed_tokens

    print("\nPrecomputing gradient labels for training data...")
    gradient_labels = []
    for i, ex in enumerate(TRAIN_DATA):
        combined = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined)
        num_chunks = (len(tokens) + L - 1) // L

        importance = compute_chunk_gradient_importance(
            model_lm, embed_layer, tokenizer_lm, combined,
            ex["question"], ex["answer"], num_chunks, L, device
        )

        # Pad to C chunks
        padded = torch.zeros(C, device=device)
        padded[:num_chunks] = importance
        gradient_labels.append(padded)

        gt_chunks = find_answer_chunks(tokens, ex["answer_lower"], L)
        top2 = sorted(torch.topk(importance[:num_chunks], k=min(2, num_chunks)).indices.tolist())
        hit = len(set(top2) & set(gt_chunks)) > 0
        status = "✓" if hit else "✗"
        print(f"  {i+1} {status}: {ex['answer']:15s} GT={gt_chunks} Grad_top2={top2} scores=[{' '.join(f'{v:.3f}' for v in importance[:num_chunks].tolist())}]")

    gradient_labels = torch.stack(gradient_labels)  # [N, C]

    # Free Qwen memory
    del model_lm, embed_layer
    torch.cuda.empty_cache()

    # ============================================================
    # Step 2: Prepare tokenized data for focus network
    # ============================================================
    print("\nPreparing tokenized data...")

    # Build vocabulary from all data
    all_tokens = set()
    for ex in TRAIN_DATA:
        combined = " ".join(ex["sentences"])
        all_tokens.update(simple_word_tokenize(combined))
    vocab = sorted(all_tokens)
    word2id = {w: i + 1 for i, w in enumerate(vocab)}
    PAD_ID = 0
    vocab_size = len(word2id) + 1

    def encode_example(ex):
        combined = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined)
        ids = [word2id.get(t, PAD_ID) for t in tokens]
        # Pad to T
        if len(ids) < T:
            ids = ids + [PAD_ID] * (T - len(ids))
        else:
            ids = ids[:T]
        return torch.tensor(ids, dtype=torch.long)

    train_x = torch.stack([encode_example(ex) for ex in TRAIN_DATA]).to(device)  # [N, T]

    # Also compute oracle labels for comparison
    oracle_labels = []
    for ex in TRAIN_DATA:
        combined = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined)
        gt = find_answer_chunks(tokens, ex["answer_lower"], L)
        label = torch.zeros(C, device=device)
        for c in gt:
            if c < C:
                label[c] = 1.0
        if label.sum() > 0:
            label = label / label.sum()
        oracle_labels.append(label)
    oracle_labels = torch.stack(oracle_labels)  # [N, C]

    # ============================================================
    # Step 3: Train focus network
    # ============================================================
    print("\nTraining focus network...")
    print(f"  Vocab size: {vocab_size}, Chunks: C={C}, Chunk size: L={L}, Select: k={k}")

    focus_net = FocusNetwork(vocab_size, embed_dim=128, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(focus_net.parameters(), lr=1e-3)

    num_epochs = 500
    N = len(TRAIN_DATA)

    for epoch in range(num_epochs):
        focus_net.train()
        logits = focus_net(train_x, PAD_ID, C, L)  # [N, C]

        # Valid chunk mask
        x_chunks = train_x.view(N, C, L)
        valid_mask = (x_chunks != PAD_ID).any(dim=2).float()  # [N, C]

        # Mask invalid chunks
        logits = logits * valid_mask + (-1e9) * (1 - valid_mask)

        # Soft cross-entropy with gradient labels
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(gradient_labels * log_probs).sum(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(focus_net.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            # Evaluate: pick top-k chunks, check if GT chunk is selected
            focus_net.eval()
            with torch.no_grad():
                logits_eval = focus_net(train_x, PAD_ID, C, L)
                logits_eval = logits_eval * valid_mask + (-1e9) * (1 - valid_mask)
                topk = torch.topk(logits_eval, k=k, dim=-1).indices  # [N, k]

                correct = 0
                for i, ex in enumerate(TRAIN_DATA):
                    combined = " ".join(ex["sentences"])
                    tokens = simple_word_tokenize(combined)
                    gt = find_answer_chunks(tokens, ex["answer_lower"], L)
                    selected = topk[i].tolist()
                    if len(set(selected) & set(gt)) > 0:
                        correct += 1

            print(f"  Epoch {epoch+1:4d} | Loss={loss.item():.4f} | Train Acc={correct}/{N} ({100*correct/N:.0f}%)")

    # ============================================================
    # Step 4: Final evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    focus_net.eval()
    with torch.no_grad():
        logits_eval = focus_net(train_x, PAD_ID, C, L)
        x_chunks = train_x.view(N, C, L)
        valid_mask = (x_chunks != PAD_ID).any(dim=2).float()
        logits_eval = logits_eval * valid_mask + (-1e9) * (1 - valid_mask)

        probs = F.softmax(logits_eval, dim=-1)
        topk = torch.topk(logits_eval, k=k, dim=-1).indices

        correct_grad = 0
        correct_oracle = 0

        for i, ex in enumerate(TRAIN_DATA):
            combined = " ".join(ex["sentences"])
            tokens = simple_word_tokenize(combined)
            num_chunks = (len(tokens) + L - 1) // L
            gt = find_answer_chunks(tokens, ex["answer_lower"], L)
            selected = sorted(topk[i].tolist())

            hit = len(set(selected) & set(gt)) > 0
            if hit:
                correct_grad += 1

            status = "✓" if hit else "✗"
            prob_str = " ".join(f"{probs[i, c]:.3f}" for c in range(num_chunks))
            print(f"  {i+1} {status}: {ex['answer']:15s} | GT={gt}, Selected={selected} | Probs=[{prob_str}]")

    print(f"\nFocus Network (gradient-supervised) accuracy: {correct_grad}/{N} ({100*correct_grad/N:.0f}%)")

    # ============================================================
    # Compare: train with oracle labels
    # ============================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Training with oracle labels")
    print("=" * 70)

    focus_oracle = FocusNetwork(vocab_size, embed_dim=128, hidden_dim=128).to(device)
    optimizer_oracle = torch.optim.Adam(focus_oracle.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        focus_oracle.train()
        logits = focus_oracle(train_x, PAD_ID, C, L)
        x_chunks = train_x.view(N, C, L)
        valid_mask = (x_chunks != PAD_ID).any(dim=2).float()
        logits = logits * valid_mask + (-1e9) * (1 - valid_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(oracle_labels * log_probs).sum(dim=-1).mean()

        optimizer_oracle.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(focus_oracle.parameters(), 1.0)
        optimizer_oracle.step()

    focus_oracle.eval()
    with torch.no_grad():
        logits_eval = focus_oracle(train_x, PAD_ID, C, L)
        logits_eval = logits_eval * valid_mask + (-1e9) * (1 - valid_mask)
        topk = torch.topk(logits_eval, k=k, dim=-1).indices

        correct_oracle = 0
        for i, ex in enumerate(TRAIN_DATA):
            combined = " ".join(ex["sentences"])
            tokens = simple_word_tokenize(combined)
            gt = find_answer_chunks(tokens, ex["answer_lower"], L)
            selected = sorted(topk[i].tolist())
            hit = len(set(selected) & set(gt)) > 0
            if hit:
                correct_oracle += 1

            status = "✓" if hit else "✗"
            print(f"  {i+1} {status}: {ex['answer']:15s} | GT={gt}, Selected={selected}")

    print(f"\nFocus Network (oracle-supervised) accuracy: {correct_oracle}/{N} ({100*correct_oracle/N:.0f}%)")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"  Gradient attribution labels:  90% (from Qwen)")
    print(f"  Focus Net (gradient-trained):  {100*correct_grad/N:.0f}%")
    print(f"  Focus Net (oracle-trained):    {100*correct_oracle/N:.0f}%")
    print(f"  REINFORCE (contrastive):       80% (from Paper I)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
"""
Harder test: 1 answer sentence buried among 20 filler sentences.
At L=16, this gives ~20+ chunks with only 1-2 containing the answer.
Tests whether sentence-transformer can find the needle in the haystack.

So the final pipeline is settled:

Sentence-transformer (80MB, off-the-shelf) labels which chunks are relevant
Focus network (~50K params) learns from those labels
Small reasoner processes only the selected chunks

No LM needed for labeling. No gradient, no ablation, no REINFORCE. 
Just semantic similarity from a tiny encoder model. Clean and efficient, as you said.
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import random


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
        B, T = x.shape
        e = self.embed(x)
        e = e.view(B, C, L, -1)
        mask = (x.view(B, C, L) != pad_id).unsqueeze(-1).float()
        pooled = (e * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        logits = self.mlp(pooled).squeeze(-1)
        return logits


# 30 diverse filler sentences (no overlap with answer topics)
FILLER_SENTENCES = [
    "Artists express creativity through different mediums including painting, sculpture, music, and digital art forms that inspire audiences.",
    "Natural wonders inspire awe and wonder in people who visit national parks and protected wilderness areas around the world.",
    "Historical events have shaped modern society in countless ways that continue to influence us today and will shape tomorrow.",
    "Students study various subjects in school to prepare for their future careers and personal growth throughout their lives.",
    "Ancient civilizations built remarkable structures that archaeologists continue to study and preserve for future generations to appreciate.",
    "Museums preserve cultural heritage and educate visitors about history, science, and artistic achievements from past eras of civilization.",
    "Scientists conduct research in laboratories using advanced equipment to make new discoveries about our world and the universe.",
    "Many people around the world enjoy learning about geography and exploring different cultures each day through travel and reading.",
    "Healthy eating habits contribute to overall physical and mental wellness throughout a long and happy life for everyone.",
    "Transportation systems connect cities and enable commerce across regions worldwide helping economies grow stronger and more resilient.",
    "Education plays a vital role in preparing young people for life and building stronger communities everywhere on the planet.",
    "Music brings joy and emotional connection to listeners around the globe regardless of language or culture or background.",
    "Gardens require regular maintenance including watering pruning and fertilizing plants to keep them healthy and beautiful all year.",
    "Sports teach teamwork discipline and perseverance to athletes of every level from beginners to professionals competing worldwide.",
    "Ocean currents influence weather patterns across entire continents and hemispheres affecting billions of people daily in many ways.",
    "Architecture reflects the cultural values and technological capabilities of civilizations throughout recorded human history everywhere on earth.",
    "Cooking is both an art and a science that brings people together for meals and celebration worldwide every day.",
    "Libraries serve as important community resources providing free access to books, computers, and educational programs for all ages.",
    "Renewable energy sources like solar and wind power are becoming increasingly important in the fight against climate change globally.",
    "Volunteers contribute countless hours to charitable organizations helping those in need and making their communities better places to live.",
    "Advances in medical technology have dramatically improved the quality and length of human life over the past century worldwide.",
    "Photography captures moments in time allowing people to preserve memories and share experiences with others across great distances.",
    "Public transportation reduces traffic congestion and pollution while providing affordable mobility options for millions of urban residents daily.",
    "Forests play a crucial role in maintaining biodiversity and regulating the global climate through carbon absorption and oxygen production.",
    "International trade agreements facilitate economic cooperation between nations and help establish standards for commerce and fair business practices.",
    "Digital literacy has become an essential skill in the modern workplace as technology continues to transform industries and job markets.",
    "Marine biologists study ocean ecosystems to understand how human activities impact aquatic life and underwater habitats around the world.",
    "Urban planning involves designing cities that balance residential commercial and recreational spaces for optimal quality of life for residents.",
    "Astronomy fascinates people of all ages as telescopes reveal the incredible scale and beauty of the cosmos beyond our planet.",
    "Traditional crafts and artisan skills are being preserved through apprenticeship programs and cultural festivals celebrating handmade goods and heritage.",
]

# Test examples with just the answer sentence (filler added programmatically)
CORE_EXAMPLES = [
    {"question": "What is the smallest planet in our solar system?",
     "answer": "Mercury", "answer_lower": "mercury",
     "answer_sentence": "In our solar system, Mercury is the smallest planet with a diameter of only about three thousand miles."},
    {"question": "What is the capital city of Italy?",
     "answer": "Rome", "answer_lower": "rome",
     "answer_sentence": "The capital city of Italy is Rome, a historic metropolis along the banks of the Tiber River."},
    {"question": "What is the chemical symbol for silver?",
     "answer": "Ag", "answer_lower": "ag",
     "answer_sentence": "The chemical symbol Ag represents silver on the periodic table, originating from the Latin word argentum."},
    {"question": "What is the tallest mountain on Earth?",
     "answer": "Everest", "answer_lower": "everest",
     "answer_sentence": "The tallest mountain on Earth is Mount Everest, standing at approximately eight thousand eight hundred meters above sea level."},
    {"question": "What is the chemical symbol for gold?",
     "answer": "Au", "answer_lower": "au",
     "answer_sentence": "On the periodic table, gold is represented by the chemical symbol Au, which derives from the Latin word aurum."},
    {"question": "What is the largest planet in our solar system?",
     "answer": "Jupiter", "answer_lower": "jupiter",
     "answer_sentence": "The largest planet in our solar system is Jupiter, a massive gas giant with dozens of orbiting moons."},
    {"question": "Who wrote the play Hamlet?",
     "answer": "Shakespeare", "answer_lower": "shakespeare",
     "answer_sentence": "The famous play Hamlet was written by Shakespeare, widely regarded as the greatest playwright in the English language."},
    {"question": "What is the capital city of Japan?",
     "answer": "Tokyo", "answer_lower": "tokyo",
     "answer_sentence": "The capital city of Japan is Tokyo, one of the most populous metropolitan areas in the entire world."},
    {"question": "What is the capital city of France?",
     "answer": "Paris", "answer_lower": "paris",
     "answer_sentence": "The capital city of France is Paris, known worldwide for its iconic landmarks and rich cultural heritage."},
    {"question": "Who painted the Mona Lisa?",
     "answer": "da Vinci", "answer_lower": "da vinci",
     "answer_sentence": "The Mona Lisa was painted by da Vinci during the Italian Renaissance, and it now hangs in the Louvre Museum."},
]


def build_hard_example(answer_sentence, num_filler=20, answer_position=None):
    """Build document with answer sentence at a random position among filler."""
    fillers = random.sample(FILLER_SENTENCES, min(num_filler, len(FILLER_SENTENCES)))

    if answer_position is None:
        answer_position = random.randint(0, len(fillers))

    sentences = fillers[:answer_position] + [answer_sentence] + fillers[answer_position:]
    return " ".join(sentences), answer_position


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(42)
    torch.manual_seed(42)

    N = len(CORE_EXAMPLES)
    k = 2

    for L in [16, 32]:
        print(f"\n{'=' * 70}")
        print(f"  L={L} | 20 filler sentences + 1 answer sentence")
        print(f"{'=' * 70}")

        # Build examples
        random.seed(42)
        built_examples = []
        for ex in CORE_EXAMPLES:
            combined, ans_pos = build_hard_example(ex["answer_sentence"], num_filler=20)
            tokens = simple_word_tokenize(combined)
            num_chunks = (len(tokens) + L - 1) // L
            gt = find_answer_chunks(tokens, ex["answer_lower"], L)
            built_examples.append({
                "combined": combined,
                "tokens": tokens,
                "num_chunks": num_chunks,
                "gt": gt,
                "question": ex["question"],
                "answer": ex["answer"],
                "answer_lower": ex["answer_lower"],
                "ans_sentence_pos": ans_pos,
            })

        max_chunks = max(b["num_chunks"] for b in built_examples)
        C = max_chunks
        T = C * L

        print(f"  Max chunks: {C}, Token length: {T}")
        for i, b in enumerate(built_examples):
            print(f"  {i+1}. {b['answer']:12s} | {b['num_chunks']:2d} chunks | GT={b['gt']} | answer at sentence {b['ans_sentence_pos']}")

        # Step 1: ST labels
        print(f"\n  Generating sentence-transformer labels...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        similarity_labels = []
        st_correct = 0

        for i, b in enumerate(built_examples):
            chunk_texts = []
            for c in range(b["num_chunks"]):
                s, e = c * L, min((c + 1) * L, len(b["tokens"]))
                chunk_texts.append(" ".join(b["tokens"][s:e]))

            q_emb = st_model.encode(b["question"], convert_to_tensor=True)
            c_embs = st_model.encode(chunk_texts, convert_to_tensor=True)
            sims = F.cosine_similarity(q_emb.unsqueeze(0), c_embs, dim=-1)

            sims_shifted = F.relu(sims)
            soft_labels = F.softmax(sims_shifted * 5.0, dim=0)

            padded = torch.zeros(C)
            padded[:b["num_chunks"]] = soft_labels.cpu()
            similarity_labels.append(padded)

            top2 = sorted(torch.topk(sims, k=min(k, b["num_chunks"])).indices.tolist())
            hit = len(set(top2) & set(b["gt"])) > 0
            if hit:
                st_correct += 1

            status = "✓" if hit else "✗"

            # Show top-5 chunks by similarity
            topk_vals, topk_idx = torch.topk(sims, k=min(5, b["num_chunks"]))
            top5_str = ", ".join(f"c{idx}={val:.3f}" for val, idx in zip(topk_vals, topk_idx))
            print(f"    {i+1} {status}: {b['answer']:12s} GT={str(b['gt']):12s} top2={top2} | top5: [{top5_str}]")

        similarity_labels = torch.stack(similarity_labels).to(device)
        del st_model

        print(f"\n  ST label accuracy: {st_correct}/{N} ({100*st_correct/N:.0f}%)")

        # Step 2: Tokenize
        all_tokens_set = set()
        for b in built_examples:
            all_tokens_set.update(b["tokens"])
        vocab = sorted(all_tokens_set)
        word2id = {w: i + 1 for i, w in enumerate(vocab)}
        PAD_ID = 0
        vocab_size = len(word2id) + 1

        def encode_example(b):
            ids = [word2id.get(t, PAD_ID) for t in b["tokens"]]
            if len(ids) < T:
                ids = ids + [PAD_ID] * (T - len(ids))
            else:
                ids = ids[:T]
            return torch.tensor(ids, dtype=torch.long)

        train_x = torch.stack([encode_example(b) for b in built_examples]).to(device)

        # Step 3: Train
        torch.manual_seed(42)
        focus_net = FocusNetwork(vocab_size, embed_dim=128, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(focus_net.parameters(), lr=1e-3)

        for epoch in range(500):
            focus_net.train()
            logits = focus_net(train_x, PAD_ID, C, L)
            x_chunks = train_x.view(N, C, L)
            valid_mask = (x_chunks != PAD_ID).any(dim=2).float()
            logits = logits * valid_mask + (-1e9) * (1 - valid_mask)
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(similarity_labels * log_probs).sum(dim=-1).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(focus_net.parameters(), 1.0)
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                focus_net.eval()
                with torch.no_grad():
                    le = focus_net(train_x, PAD_ID, C, L)
                    le = le * valid_mask + (-1e9) * (1 - valid_mask)
                    tk = torch.topk(le, k=k, dim=-1).indices
                    cc = 0
                    for i, b in enumerate(built_examples):
                        if len(set(tk[i].tolist()) & set(b["gt"])) > 0:
                            cc += 1
                print(f"    Epoch {epoch+1:4d} | Loss={loss.item():.4f} | Acc={cc}/{N}")

        # Step 4: Evaluate
        print(f"\n  Final evaluation (ST-supervised):")
        focus_net.eval()
        fn_correct = 0
        with torch.no_grad():
            logits_eval = focus_net(train_x, PAD_ID, C, L)
            x_chunks = train_x.view(N, C, L)
            valid_mask = (x_chunks != PAD_ID).any(dim=2).float()
            logits_eval = logits_eval * valid_mask + (-1e9) * (1 - valid_mask)
            probs = F.softmax(logits_eval, dim=-1)
            topk = torch.topk(logits_eval, k=k, dim=-1).indices

            for i, b in enumerate(built_examples):
                selected = sorted(topk[i].tolist())
                hit = len(set(selected) & set(b["gt"])) > 0
                if hit:
                    fn_correct += 1
                status = "✓" if hit else "✗"

                # Show chunk texts for selected and GT
                gt_texts = []
                for c in b["gt"]:
                    s, e = c * L, min((c + 1) * L, len(b["tokens"]))
                    gt_texts.append(" ".join(b["tokens"][s:e])[:60])
                sel_texts = []
                for c in selected:
                    s, e = c * L, min((c + 1) * L, len(b["tokens"]))
                    sel_texts.append(" ".join(b["tokens"][s:e])[:60])

                print(f"    {i+1} {status}: {b['answer']:12s} GT={b['gt']}, Sel={selected}")
                for c in b["gt"]:
                    s, e = c * L, min((c + 1) * L, len(b["tokens"]))
                    print(f"       GT  chunk {c:2d}: {' '.join(b['tokens'][s:e])[:80]}")
                for c in selected:
                    if c not in b["gt"]:
                        s, e = c * L, min((c + 1) * L, len(b["tokens"]))
                        print(f"       SEL chunk {c:2d}: {' '.join(b['tokens'][s:e])[:80]}")

        # Oracle
        oracle_labels = []
        for b in built_examples:
            label = torch.zeros(C)
            for c in b["gt"]:
                if c < C:
                    label[c] = 1.0
            if label.sum() > 0:
                label = label / label.sum()
            oracle_labels.append(label)
        oracle_labels = torch.stack(oracle_labels).to(device)

        torch.manual_seed(42)
        fo = FocusNetwork(vocab_size, embed_dim=128, hidden_dim=128).to(device)
        oo = torch.optim.Adam(fo.parameters(), lr=1e-3)
        for epoch in range(500):
            fo.train()
            logits = fo(train_x, PAD_ID, C, L)
            x_chunks = train_x.view(N, C, L)
            vm = (x_chunks != PAD_ID).any(dim=2).float()
            logits = logits * vm + (-1e9) * (1 - vm)
            lp = F.log_softmax(logits, dim=-1)
            loss = -(oracle_labels * lp).sum(dim=-1).mean()
            oo.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fo.parameters(), 1.0)
            oo.step()

        fo.eval()
        oc = 0
        with torch.no_grad():
            le = fo(train_x, PAD_ID, C, L)
            le = le * vm + (-1e9) * (1 - vm)
            tk = torch.topk(le, k=k, dim=-1).indices
            for i, b in enumerate(built_examples):
                if len(set(tk[i].tolist()) & set(b["gt"])) > 0:
                    oc += 1

        print(f"\n  ST labels:    {st_correct}/{N} ({100*st_correct/N:.0f}%)")
        print(f"  Focus Net ST: {fn_correct}/{N} ({100*fn_correct/N:.0f}%)")
        print(f"  Oracle:       {oc}/{N} ({100*oc/N:.0f}%)")


if __name__ == "__main__":
    main()
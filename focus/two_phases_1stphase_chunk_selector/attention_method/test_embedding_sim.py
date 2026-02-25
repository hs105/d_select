"""
Embedding similarity: use Qwen as an encoder (not generator).
- Encode each chunk -> mean-pool hidden states -> chunk embedding
- Encode the question -> question embedding
- Cosine similarity -> most similar chunk is most relevant

No generation, no logP, no parametric knowledge problem.
"""
import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def get_embedding(model, tokenizer, text, device):
    """Encode text, mean-pool last hidden states."""
    ids = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
        embedding = hidden.mean(dim=0)  # [hidden_dim]
    return embedding


TEST_EXAMPLES = [
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Qwen as encoder...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True,
    )
    model.eval()

    L = 8
    k = 2
    correct = 0
    total = len(TEST_EXAMPLES)

    print(f"\n{'=' * 70}")
    print("EMBEDDING SIMILARITY: Cosine(chunk, question)")
    print(f"{'=' * 70}")

    for i, ex in enumerate(TEST_EXAMPLES):
        combined = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined)
        num_chunks = (len(tokens) + L - 1) // L
        gt_chunks = find_answer_chunks(tokens, ex["answer_lower"], L)

        # Embed the question
        q_emb = get_embedding(model, tokenizer, ex["question"], device)

        # Embed each chunk
        chunk_sims = []
        chunk_texts = []
        for c in range(num_chunks):
            s, e = c * L, min((c + 1) * L, len(tokens))
            chunk_text = " ".join(tokens[s:e])
            chunk_texts.append(chunk_text)

            c_emb = get_embedding(model, tokenizer, chunk_text, device)
            sim = F.cosine_similarity(q_emb.unsqueeze(0), c_emb.unsqueeze(0)).item()
            chunk_sims.append(sim)

        chunk_sims_t = torch.tensor(chunk_sims)

        # Top-2
        top2 = sorted(torch.topk(chunk_sims_t, k=min(k, num_chunks)).indices.tolist())
        hit = len(set(top2) & set(gt_chunks)) > 0
        if hit:
            correct += 1

        status = "✓" if hit else "✗"
        print(f"\n  {i+1} {status}: {ex['answer']:15s} | GT={gt_chunks}, Sim top2={top2}")

        # Normalize for display
        sims_shifted = chunk_sims_t - chunk_sims_t.min()
        sims_norm = sims_shifted / sims_shifted.sum().clamp_min(1e-8)

        for c in range(num_chunks):
            marker = " <<<" if c in gt_chunks else ""
            bar = "█" * int(sims_norm[c].item() * 50)
            print(f"     Chunk {c}: sim={chunk_sims[c]:.4f} {bar:20s} {chunk_texts[c]}{marker}")

    print(f"\n{'=' * 70}")
    print(f"Embedding similarity accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
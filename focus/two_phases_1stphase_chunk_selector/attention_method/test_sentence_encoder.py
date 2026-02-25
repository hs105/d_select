"""
Use a sentence-transformer (designed for similarity) instead of a causal LM.
all-MiniLM-L6-v2 is 80MB, trained specifically for cosine similarity.
"""
import re
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


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

# Filler for large-chunk tests
FILLER = (
    "People enjoy walking in parks and gardens every morning when the sun rises over the horizon. "
    "Technology continues to evolve and reshape how we communicate daily across the globe and beyond. "
    "Books provide knowledge and entertainment for readers of all ages from every corner of the world. "
    "Healthy eating habits contribute to overall physical and mental wellness throughout a long and happy life. "
    "Transportation systems connect cities and enable commerce across regions worldwide helping economies grow stronger. "
    "Education plays a vital role in preparing young people for life and building stronger communities everywhere. "
    "Music brings joy and emotional connection to listeners around the globe regardless of language or culture. "
    "Gardens require regular maintenance including watering pruning and fertilizing plants to keep them healthy and beautiful. "
    "Sports teach teamwork discipline and perseverance to athletes of every level from beginners to professionals. "
    "Ocean currents influence weather patterns across entire continents and hemispheres affecting billions of people daily. "
)


def build_document(answer_sentence, target_word_count, answer_position="middle"):
    filler_tokens = simple_word_tokenize(FILLER)
    answer_words = simple_word_tokenize(answer_sentence)
    filler_needed = target_word_count - len(answer_words)
    if filler_needed < 0:
        filler_needed = 0
    while len(filler_tokens) < filler_needed:
        filler_tokens = filler_tokens + filler_tokens

    if answer_position == "start":
        combined = answer_sentence + " " + " ".join(filler_tokens[:filler_needed])
    elif answer_position == "end":
        combined = " ".join(filler_tokens[:filler_needed]) + " " + answer_sentence
    else:
        half = filler_needed // 2
        combined = " ".join(filler_tokens[:half]) + " " + answer_sentence + " " + " ".join(filler_tokens[half:filler_needed])
    return combined


def main():
    print("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    L = 8
    k = 2

    # ============================================================
    # Test 1: Small chunks (L=8), same as previous experiments
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"TEST 1: Sentence-transformer similarity (L={L})")
    print(f"{'=' * 70}")

    correct = 0
    total = len(TEST_EXAMPLES)

    for i, ex in enumerate(TEST_EXAMPLES):
        combined = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined)
        num_chunks = (len(tokens) + L - 1) // L
        gt_chunks = find_answer_chunks(tokens, ex["answer_lower"], L)

        # Build chunk texts
        chunk_texts = []
        for c in range(num_chunks):
            s, e = c * L, min((c + 1) * L, len(tokens))
            chunk_texts.append(" ".join(tokens[s:e]))

        # Encode question and chunks
        q_emb = model.encode(ex["question"], convert_to_tensor=True)
        c_embs = model.encode(chunk_texts, convert_to_tensor=True)

        # Cosine similarity
        sims = F.cosine_similarity(q_emb.unsqueeze(0), c_embs, dim=-1)

        top2 = sorted(torch.topk(sims, k=min(k, num_chunks)).indices.tolist())
        hit = len(set(top2) & set(gt_chunks)) > 0
        if hit:
            correct += 1

        status = "✓" if hit else "✗"
        print(f"\n  {i+1} {status}: {ex['answer']:15s} | GT={gt_chunks}, top2={top2}")

        sims_shifted = sims - sims.min()
        sims_norm = sims_shifted / sims_shifted.sum().clamp_min(1e-8)
        for c in range(num_chunks):
            marker = " <<<" if c in gt_chunks else ""
            bar = "█" * int(sims_norm[c].item() * 50)
            print(f"     Chunk {c}: sim={sims[c]:.4f} {bar:20s} {chunk_texts[c]}{marker}")

    print(f"\n  Accuracy (L={L}): {correct}/{total} ({100*correct/total:.1f}%)")

    # ============================================================
    # Test 2: Larger chunks (L=16, 32) with filler
    # ============================================================
    CORE = [
        {"answer_sentence": "The capital city of France is Paris.",
         "question": "What is the capital city of France?", "answer": "Paris", "answer_lower": "paris"},
        {"answer_sentence": "The largest planet in our solar system is Jupiter.",
         "question": "What is the largest planet in our solar system?", "answer": "Jupiter", "answer_lower": "jupiter"},
        {"answer_sentence": "The play Hamlet was written by Shakespeare.",
         "question": "Who wrote the play Hamlet?", "answer": "Shakespeare", "answer_lower": "shakespeare"},
        {"answer_sentence": "The Mona Lisa was painted by da Vinci.",
         "question": "Who painted the Mona Lisa?", "answer": "da Vinci", "answer_lower": "da vinci"},
        {"answer_sentence": "The capital of Japan is Tokyo.",
         "question": "What is the capital city of Japan?", "answer": "Tokyo", "answer_lower": "tokyo"},
    ]

    for L_test in [16, 32]:
        print(f"\n{'=' * 70}")
        print(f"TEST 2: Larger chunks (L={L_test}), answer in middle of ~128-word doc")
        print(f"{'=' * 70}")

        correct_large = 0
        for ex in CORE:
            combined = build_document(ex["answer_sentence"], target_word_count=128, answer_position="middle")
            tokens = simple_word_tokenize(combined)
            num_chunks = (len(tokens) + L_test - 1) // L_test
            gt_chunks = find_answer_chunks(tokens, ex["answer_lower"], L_test)

            chunk_texts = []
            for c in range(num_chunks):
                s, e = c * L_test, min((c + 1) * L_test, len(tokens))
                chunk_texts.append(" ".join(tokens[s:e]))

            q_emb = model.encode(ex["question"], convert_to_tensor=True)
            c_embs = model.encode(chunk_texts, convert_to_tensor=True)
            sims = F.cosine_similarity(q_emb.unsqueeze(0), c_embs, dim=-1)

            top2 = sorted(torch.topk(sims, k=min(k, num_chunks)).indices.tolist())
            hit = len(set(top2) & set(gt_chunks)) > 0
            if hit:
                correct_large += 1

            status = "✓" if hit else "✗"
            print(f"\n  {status} {ex['answer']:15s} | GT={gt_chunks}, top2={top2}")
            sims_shifted = sims - sims.min()
            sims_norm = sims_shifted / sims_shifted.sum().clamp_min(1e-8)
            for c in range(num_chunks):
                marker = " <<<" if c in gt_chunks else ""
                bar = "█" * int(sims_norm[c].item() * 50)
                print(f"     Chunk {c}: {sims[c]:.4f} {bar:20s} {chunk_texts[c][:80]}{marker}")

        print(f"\n  Accuracy (L={L_test}): {correct_large}/{len(CORE)}")

    # ============================================================
    # Test 3: The Paris + filler problem
    # ============================================================
    print(f"\n{'=' * 70}")
    print("TEST 3: Paris with increasing filler (single chunk)")
    print(f"{'=' * 70}")

    question = "What is the capital city of France?"
    q_emb = model.encode(question, convert_to_tensor=True)

    texts = [
        "The capital city of France is Paris.",
        "The capital city of France is Paris. This is nice.",
        "The capital city of France is Paris. This is nice. The weather is good today.",
        "The capital city of France is Paris. This is nice. The weather is good today. People enjoy walking in parks and gardens every morning.",
        "The capital city of France is Paris. This is nice. The weather is good today. People enjoy walking in parks and gardens every morning. Technology continues to evolve and reshape how we communicate daily across the globe.",
    ]

    for text in texts:
        emb = model.encode(text, convert_to_tensor=True)
        sim = F.cosine_similarity(q_emb.unsqueeze(0), emb.unsqueeze(0)).item()
        print(f"  sim={sim:.4f} | {text[:90]}...")


if __name__ == "__main__":
    main()
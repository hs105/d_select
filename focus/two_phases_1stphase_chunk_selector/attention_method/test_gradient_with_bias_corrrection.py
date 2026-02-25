"""
Gradient attribution with positional bias correction.
1. Compute gradient importance for the actual example
2. Compute gradient importance for a baseline (random/irrelevant text of same length)
3. Subtract baseline to remove positional bias
"""
import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
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
            if chunk_toks[j:j+answer_len] == answer_tokens:
                gt_chunks.append(c)
                break
    return gt_chunks


def compute_gradient_importance(model, embed_layer, tokenizer, full_text, context_text, pred_pos_input, device):
    """Compute per-token gradient importance for a given input."""
    full_ids = tokenizer(full_text, return_tensors="pt").to(device)
    input_ids = full_ids.input_ids
    
    embeddings = embed_layer(input_ids).detach().requires_grad_(True)
    outputs = model(inputs_embeds=embeddings)
    
    # Use pred_pos from the caller
    context_ids = tokenizer(context_text, return_tensors="pt").to(device)
    context_len = context_ids.input_ids.shape[1]
    pred_pos = context_len - 1
    target_token_id = input_ids[0, context_len]
    
    logits = outputs.logits[0, pred_pos, :]
    log_probs = F.log_softmax(logits, dim=-1)
    target_logp = log_probs[target_token_id]
    
    target_logp.backward()
    
    grad = embeddings.grad[0]
    token_importance = grad.norm(dim=-1)
    
    embeddings.grad = None
    model.zero_grad()
    
    return token_importance, context_len


def map_to_chunks(token_importance, tokenizer, combined, num_chunks, device):
    """Map BPE-level importance to word-level chunks."""
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
    
    return chunk_importance


# Filler sentences for building positional baselines
FILLER_SENTENCES = [
    "The weather changes frequently throughout the seasons each year.",
    "People enjoy spending time outdoors when the sun is shining brightly.",
    "Technology continues to evolve and reshape how we communicate daily.",
    "Books provide knowledge and entertainment for readers of all ages.",
    "Healthy eating habits contribute to overall physical and mental wellness.",
    "Transportation systems connect cities and enable commerce across regions worldwide.",
    "Education plays a vital role in preparing young people for life.",
    "Music brings joy and emotional connection to listeners around the globe.",
    "Gardens require regular maintenance including watering pruning and fertilizing plants.",
    "Sports teach teamwork discipline and perseverance to athletes of every level.",
    "Ocean currents influence weather patterns across entire continents and hemispheres.",
    "Architecture reflects the cultural values and technological capabilities of civilizations.",
]


def compute_positional_baseline(model, embed_layer, tokenizer, question, answer, num_chunks, L, device, n_baselines=5):
    """
    Compute average chunk importance using irrelevant filler text.
    This captures the positional bias independent of content.
    """
    baseline_importance = torch.zeros(num_chunks, device=device)
    
    for b in range(n_baselines):
        # Shuffle filler sentences to create random context of similar length
        shuffled = random.sample(FILLER_SENTENCES, len(FILLER_SENTENCES))
        filler_combined = " ".join(shuffled)
        
        # Truncate/pad to roughly match the target number of chunks
        filler_tokens = simple_word_tokenize(filler_combined)
        target_len = num_chunks * L
        if len(filler_tokens) > target_len:
            filler_tokens = filler_tokens[:target_len]
        filler_combined = " ".join(filler_tokens)
        
        filler_num_chunks = (len(filler_tokens) + L - 1) // L
        
        input_text = filler_combined + " " + question
        full_text = input_text + " " + answer
        
        token_importance, context_len = compute_gradient_importance(
            model, embed_layer, tokenizer, full_text, input_text, None, device
        )
        
        chunk_imp = map_to_chunks(token_importance, tokenizer, filler_combined, filler_num_chunks, device)
        
        # Pad or truncate to match target num_chunks
        min_c = min(filler_num_chunks, num_chunks)
        baseline_importance[:min_c] += chunk_imp[:min_c]
    
    baseline_importance /= n_baselines
    return baseline_importance


def run_corrected_attribution(model, tokenizer, embed_layer, combined, question, answer, answer_lower, L, device, positional_baseline=None):
    tokens = simple_word_tokenize(combined)
    num_chunks = (len(tokens) + L - 1) // L
    gt_chunks = find_answer_chunks(tokens, answer_lower, L)
    
    input_text = combined + " " + question
    answer_text = " " + answer
    full_text = input_text + answer_text
    
    # Compute actual gradient importance
    token_importance, context_len = compute_gradient_importance(
        model, embed_layer, tokenizer, full_text, input_text, None, device
    )
    
    chunk_importance = map_to_chunks(token_importance, tokenizer, combined, num_chunks, device)
    
    # Compute positional baseline if not provided
    if positional_baseline is None:
        positional_baseline = compute_positional_baseline(
            model, embed_layer, tokenizer, question, answer_text, num_chunks, L, device
        )
    
    # Subtract positional bias
    corrected = chunk_importance - positional_baseline[:num_chunks]
    
    # Normalize for display
    raw_norm = chunk_importance[:num_chunks] / chunk_importance[:num_chunks].sum().clamp_min(1e-8)
    corrected_pos = F.relu(corrected[:num_chunks])  # zero out negatives
    corrected_norm = corrected_pos / corrected_pos.sum().clamp_min(1e-8)
    
    top2_raw = sorted(torch.topk(chunk_importance[:num_chunks], k=min(2, num_chunks)).indices.tolist())
    top2_corr = sorted(torch.topk(corrected[:num_chunks], k=min(2, num_chunks)).indices.tolist())
    
    hit_raw = len(set(top2_raw) & set(gt_chunks)) > 0
    hit_corr = len(set(top2_corr) & set(gt_chunks)) > 0
    
    status_raw = "✓" if hit_raw else "✗"
    status_corr = "✓" if hit_corr else "✗"
    
    print(f"  Raw   {status_raw}: {answer:15s} | GT={gt_chunks}, top2={top2_raw}")
    print(f"  Corr. {status_corr}: {answer:15s} | GT={gt_chunks}, top2={top2_corr}")
    
    for c in range(num_chunks):
        start_tok = c * L
        end_tok = min(start_tok + L, len(tokens))
        chunk_text = " ".join(tokens[start_tok:end_tok])
        marker = " <<<" if c in gt_chunks else ""
        bar_raw = "█" * int(raw_norm[c].item() * 40)
        bar_corr = "█" * int(corrected_norm[c].item() * 40)
        print(f"     Chunk {c}: raw={raw_norm[c]:.3f} {bar_raw:18s} corr={corrected_norm[c]:.3f} {bar_corr:18s} {chunk_text}{marker}")
    
    return hit_raw, hit_corr


TEST_EXAMPLES = [
    {
        "question": "What is the smallest planet in our solar system?",
        "answer": "Mercury", "answer_lower": "mercury",
        "sentences": [
            "Artists express creativity through different mediums including painting, sculpture, music, and digital art forms.",
            "Natural wonders inspire awe and wonder in people who visit national parks and protected wilderness areas.",
            "In our solar system, Mercury is the smallest planet with a diameter of only three thousand miles.",
            "The smallest planet in our solar system is Mercury.",
        ],
    },
    {
        "question": "What is the capital city of Italy?",
        "answer": "Rome", "answer_lower": "rome",
        "sentences": [
            "In Rome, the capital, you can explore ruins from the Roman Empire that still stand today.",
            "Natural wonders inspire awe and wonder in people who visit national parks and protected wilderness areas.",
            "Artists express creativity through different mediums including painting, sculpture, music, and digital art forms.",
            "The capital city of Italy is Rome.",
        ],
    },
    {
        "question": "What is the chemical symbol for silver?",
        "answer": "Ag", "answer_lower": "ag",
        "sentences": [
            "Historical events have shaped modern society in countless ways that continue to influence us today.",
            "Ancient civilizations built remarkable structures that archaeologists continue to study and preserve for future generations.",
            "The chemical symbol Ag represents silver in equations, originating from ancient Latin terminology for the metal.",
            "The chemical symbol for silver is Ag.",
        ],
    },
    {
        "question": "What is the tallest mountain on Earth?",
        "answer": "Everest", "answer_lower": "everest",
        "sentences": [
            "Students study various subjects in school to prepare for their future careers and personal growth.",
            "The highest point on Earth is the peak of Mount Everest, which towers above all other mountains.",
            "Historical events have shaped modern society in countless ways that continue to influence us today.",
            "The tallest mountain on Earth is Mount Everest.",
        ],
    },
    {
        "question": "What is the chemical symbol for gold?",
        "answer": "Au", "answer_lower": "au",
        "sentences": [
            "Students study various subjects in school to prepare for their future careers and personal growth.",
            "Museums preserve cultural heritage and educate visitors about history, science, and artistic achievements from past eras.",
            "On the periodic table, gold is represented by the symbol Au, derived from the Latin word aurum.",
            "The chemical symbol for gold is Au.",
        ],
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter", "answer_lower": "jupiter",
        "sentences": [
            "Scientists conduct research in laboratories using advanced equipment to make new discoveries about our world.",
            "Astronomers study Jupiter because it is the biggest planet and has dozens of interesting moons orbiting it.",
            "Ancient civilizations built remarkable structures that archaeologists continue to study and preserve for future generations.",
            "The largest planet in our solar system is Jupiter.",
        ],
    },
    {
        "question": "Who wrote the play Hamlet?",
        "answer": "Shakespeare", "answer_lower": "shakespeare",
        "sentences": [
            "Theater companies worldwide perform Hamlet by Shakespeare, who is considered the greatest English playwright ever.",
            "Students study various subjects in school to prepare for their future careers and personal growth.",
            "Ancient civilizations built remarkable structures that archaeologists continue to study and preserve for future generations.",
            "The play Hamlet was written by Shakespeare.",
        ],
    },
    {
        "question": "What is the capital city of Japan?",
        "answer": "Tokyo", "answer_lower": "tokyo",
        "sentences": [
            "Historical events have shaped modern society in countless ways that continue to influence us today.",
            "Scientists conduct research in laboratories using advanced equipment to make new discoveries about our world.",
            "Mount Fuji can be seen from Tokyo on clear days, though the capital is quite far from it.",
            "The capital of Japan is Tokyo.",
        ],
    },
    {
        "question": "What is the capital city of France?",
        "answer": "Paris", "answer_lower": "paris",
        "sentences": [
            "In a trivia quiz, the capital of France is Paris, and many tourists visit Paris every year.",
            "Many people around the world enjoy learning about geography and exploring different cultures each day.",
            "Museums preserve cultural heritage and educate visitors about history, science, and artistic achievements from past eras.",
            "The capital city of France is Paris. Paris is a great place.",
        ],
    },
    {
        "question": "Who painted the Mona Lisa?",
        "answer": "da Vinci", "answer_lower": "da vinci",
        "sentences": [
            "Students study various subjects in school to prepare for their future careers and personal growth.",
            "The artist da Vinci spent years perfecting the Mona Lisa, working on details with incredible precision and care.",
            "Historical events have shaped modern society in countless ways that continue to influence us today.",
            "The Mona Lisa was painted by da Vinci.",
        ],
    },
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(42)
    
    print("Loading Qwen...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        device_map="auto",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    embed_layer = model.model.embed_tokens
    
    L = 8
    raw_correct = 0
    corr_correct = 0
    total = len(TEST_EXAMPLES)
    
    print("\n" + "=" * 80)
    print("GRADIENT ATTRIBUTION: Raw vs Positional-Bias-Corrected")
    print("=" * 80)
    
    for i, ex in enumerate(TEST_EXAMPLES):
        combined = " ".join(ex["sentences"])
        print(f"\n--- Example {i+1} ---")
        hit_raw, hit_corr = run_corrected_attribution(
            model, tokenizer, embed_layer, combined,
            ex["question"], ex["answer"], ex["answer_lower"],
            L, device
        )
        if hit_raw:
            raw_correct += 1
        if hit_corr:
            corr_correct += 1
    
    print(f"\n{'=' * 80}")
    print(f"Raw gradient accuracy:       {raw_correct}/{total} ({100*raw_correct/total:.1f}%)")
    print(f"Corrected gradient accuracy:  {corr_correct}/{total} ({100*corr_correct/total:.1f}%)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
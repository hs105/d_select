"""
Ablation attribution: zero out one chunk at a time, measure logP(answer) drop.

For each chunk c:
  1. Replace chunk c's embeddings with zeros
  2. Compute logP(answer | context_with_chunk_c_zeroed, question)
  3. Importance(c) = logP(full) - logP(without c)

The chunk whose removal causes the biggest drop is the most important.

Suffer from the same problem: model already knows this and does not needs input

I hate this model uses the knowledge it learned elsewhere to use here: i want it to focus on the input only and and tell me which chunk has the correct information. 
This seems a very simple task, yet so much frustration. 
How do we solve this problem? 

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


def compute_logp(model, embeddings, pred_pos, target_token_id):
    """Forward pass on embeddings, return logP of target token at pred_pos."""
    with torch.no_grad():
        outputs = model(inputs_embeds=embeddings)
        logits = outputs.logits[0, pred_pos, :]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[target_token_id].item()


def ablation_attribution(model, embed_layer, tokenizer, combined, question, answer, L, device):
    """
    Zero out one chunk at a time, measure logP drop.
    Returns (chunk_importance, num_chunks, logp_full).
    """
    input_text = combined + " " + question
    full_text = input_text + " " + answer

    context_ids = tokenizer(input_text, return_tensors="pt").to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").to(device)
    context_len = context_ids.input_ids.shape[1]

    input_ids = full_ids.input_ids
    pred_pos = context_len - 1
    target_token_id = input_ids[0, context_len]

    # Get full embeddings
    full_embeds = embed_layer(input_ids).detach()  # [1, seq_len, hidden]

    # Get context BPE length for chunk mapping
    context_only_ids = tokenizer(combined, return_tensors="pt").to(device)
    context_bpe_len = context_only_ids.input_ids.shape[1]

    tokens = simple_word_tokenize(combined)
    num_chunks = (len(tokens) + L - 1) // L

    # Map each BPE position to a chunk index
    bpe_to_chunk = []
    for bpe_pos in range(context_bpe_len):
        frac = bpe_pos / max(context_bpe_len, 1)
        chunk_idx = min(int(frac * num_chunks), num_chunks - 1)
        bpe_to_chunk.append(chunk_idx)

    # Baseline: logP with full context
    logp_full = compute_logp(model, full_embeds, pred_pos, target_token_id)

    # Ablate each chunk
    chunk_importance = torch.zeros(num_chunks)
    for c in range(num_chunks):
        ablated_embeds = full_embeds.clone()
        # Zero out BPE positions belonging to chunk c
        for bpe_pos in range(context_bpe_len):
            if bpe_to_chunk[bpe_pos] == c:
                ablated_embeds[0, bpe_pos, :] = 0.0

        logp_ablated = compute_logp(model, ablated_embeds, pred_pos, target_token_id)
        # Importance = how much logP drops when we remove this chunk
        chunk_importance[c] = logp_full - logp_ablated

    return chunk_importance, num_chunks, logp_full


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

    print("Loading Qwen...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", device_map="auto", torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()
    embed_layer = model.model.embed_tokens

    L = 8
    k = 2
    correct = 0
    total = len(TEST_EXAMPLES)

    print(f"\n{'=' * 70}")
    print("ABLATION ATTRIBUTION: Zero out one chunk at a time")
    print(f"{'=' * 70}")

    for i, ex in enumerate(TEST_EXAMPLES):
        combined = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined)
        gt_chunks = find_answer_chunks(tokens, ex["answer_lower"], L)

        chunk_imp, num_chunks, logp_full = ablation_attribution(
            model, embed_layer, tokenizer, combined,
            ex["question"], ex["answer"], L, device
        )

        # Top-2 by importance (biggest logP drop)
        top2 = sorted(torch.topk(chunk_imp[:num_chunks], k=min(k, num_chunks)).indices.tolist())
        hit = len(set(top2) & set(gt_chunks)) > 0
        if hit:
            correct += 1

        status = "✓" if hit else "✗"
        print(f"\n  {i+1} {status}: {ex['answer']:15s} | GT={gt_chunks}, Ablation top2={top2} | logP_full={logp_full:.3f}")

        # Normalize for display
        imp_pos = F.relu(chunk_imp[:num_chunks])
        imp_norm = imp_pos / imp_pos.sum().clamp_min(1e-8)

        for c in range(num_chunks):
            s, e = c * L, min((c + 1) * L, len(tokens))
            text = " ".join(tokens[s:e])
            marker = " <<<" if c in gt_chunks else ""
            bar = "█" * int(imp_norm[c].item() * 50)
            drop_str = f"{chunk_imp[c]:+.3f}"
            print(f"     Chunk {c}: drop={drop_str:8s} {imp_norm[c]:.3f} {bar:20s} {text}{marker}")

    print(f"\n{'=' * 70}")
    print(f"Ablation attribution accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
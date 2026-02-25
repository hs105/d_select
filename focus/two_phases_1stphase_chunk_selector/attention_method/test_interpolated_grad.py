"""
Integrated Gradients attribution for chunk importance.
More stable than raw gradients by integrating along the path
from zero baseline to actual embeddings.
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
            if chunk_toks[j:j+answer_len] == answer_tokens:
                gt_chunks.append(c)
                break
    return gt_chunks


def integrated_gradients(model, embed_layer, input_ids, pred_pos, target_token_id, device, n_steps=30):
    """
    Compute integrated gradients from zero baseline to actual embeddings.
    
    1. baseline = zero embeddings
    2. actual = real embeddings
    3. For alpha in [0, 1], compute gradient at baseline + alpha * (actual - baseline)
    4. Average gradients, multiply by (actual - baseline)
    """
    actual_embeds = embed_layer(input_ids).detach()  # [1, seq_len, hidden]
    baseline_embeds = torch.zeros_like(actual_embeds)  # zero baseline
    
    # Accumulate gradients along the path
    accumulated_grads = torch.zeros_like(actual_embeds)
    
    for step in range(n_steps):
        alpha = step / n_steps
        
        # Interpolated input
        interp_embeds = baseline_embeds + alpha * (actual_embeds - baseline_embeds)
        interp_embeds = interp_embeds.detach().requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs_embeds=interp_embeds)
        logits = outputs.logits[0, pred_pos, :]
        log_probs = F.log_softmax(logits, dim=-1)
        target_logp = log_probs[target_token_id]
        
        # Backward
        target_logp.backward()
        
        accumulated_grads += interp_embeds.grad.detach()
        
        interp_embeds.grad = None
        model.zero_grad()
    
    # Average gradients and multiply by (actual - baseline)
    avg_grads = accumulated_grads / n_steps
    integrated_grads = avg_grads * (actual_embeds - baseline_embeds)
    
    # Per-token importance: sum absolute values across hidden dim
    token_importance = integrated_grads[0].abs().sum(dim=-1)  # [seq_len]
    
    return token_importance


def run_ig_attribution(model, tokenizer, embed_layer, combined, question, answer, answer_lower, L, device, n_steps=30):
    tokens = simple_word_tokenize(combined)
    num_chunks = (len(tokens) + L - 1) // L
    gt_chunks = find_answer_chunks(tokens, answer_lower, L)
    
    input_text = combined + " " + question
    answer_text = " " + answer
    full_text = input_text + answer_text
    
    context_ids = tokenizer(input_text, return_tensors="pt").to(device)
    full_ids = tokenizer(full_text, return_tensors="pt").to(device)
    
    context_len = context_ids.input_ids.shape[1]
    
    input_ids = full_ids.input_ids
    pred_pos = context_len - 1
    target_token_id = input_ids[0, context_len]
    
    # Compute integrated gradients
    token_importance = integrated_gradients(
        model, embed_layer, input_ids, pred_pos, target_token_id, device, n_steps=n_steps
    )
    
    # Map BPE tokens to word-level chunks
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
    chunk_imp_norm = chunk_importance[:num_chunks] / chunk_importance[:num_chunks].sum().clamp_min(1e-8)
    
    top2 = sorted(torch.topk(chunk_importance[:num_chunks], k=min(2, num_chunks)).indices.tolist())
    hit = len(set(top2) & set(gt_chunks)) > 0
    
    status = "✓" if hit else "✗"
    print(f"  {status}: {answer:15s} | GT={gt_chunks}, IG top2={top2}")
    
    for c in range(num_chunks):
        start_tok = c * L
        end_tok = min(start_tok + L, len(tokens))
        chunk_text = " ".join(tokens[start_tok:end_tok])
        marker = " <<<" if c in gt_chunks else ""
        bar = "█" * int(chunk_imp_norm[c].item() * 50)
        print(f"     Chunk {c}: {chunk_imp_norm[c]:.3f} {bar:20s} {chunk_text}{marker}")
    
    model.zero_grad()
    return hit


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
    n_steps = 30  # number of interpolation steps
    correct = 0
    total = len(TEST_EXAMPLES)
    
    print(f"\nUsing {n_steps} interpolation steps for Integrated Gradients")
    print("=" * 70)
    print("INTEGRATED GRADIENTS ATTRIBUTION")
    print("=" * 70)
    
    for i, ex in enumerate(TEST_EXAMPLES):
        combined = " ".join(ex["sentences"])
        print(f"\n--- Example {i+1} ---")
        hit = run_ig_attribution(
            model, tokenizer, embed_layer, combined,
            ex["question"], ex["answer"], ex["answer_lower"],
            L, device, n_steps=n_steps
        )
        if hit:
            correct += 1
    
    print(f"\n{'=' * 70}")
    print(f"Integrated Gradients accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
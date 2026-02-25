"""
Test: Does gradient attribution degrade with larger chunks + trailing filler?

We test with a single example at different chunk sizes and with different
amounts of filler after the answer word to understand what gradient actually measures.


Note: this shows Gradient: leaks to positions after the answer. 
The gradient leaks through the residual stream to subsequent tokens.

Let me think about what actually works. The fundamental issue is:

We want to know: "which chunk contains the answer?"
The LM says: "I don't care, I already know the answer"

Maybe the question is wrong. 
Instead of asking the LM "do you need this chunk?", 
we should ask a model that doesn't know the answer and can only figure it out by 
reading the chunks. That's your dedicated small retrieval model point.

"""
import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def simple_word_tokenize(text):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())


def compute_token_level_gradient(model, embed_layer, tokenizer, combined, question, answer, device):
    """Return per-BPE-token gradient importance and the BPE tokens for display."""
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
    token_importance = grad.norm(dim=-1).detach()

    # Get context-only portion
    context_only_ids = tokenizer(combined, return_tensors="pt").to(device)
    context_bpe_len = context_only_ids.input_ids.shape[1]

    # Decode individual BPE tokens for display
    bpe_tokens = [tokenizer.decode([input_ids[0, j]]) for j in range(context_bpe_len)]

    embeddings.grad = None
    model.zero_grad()

    return token_importance[:context_bpe_len], bpe_tokens


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Qwen...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", device_map="auto", torch_dtype=torch.float32, trust_remote_code=True,
    )
    model.eval()
    embed_layer = model.model.embed_tokens

    question = "What is the capital city of France?"
    answer = "Paris"

    # ============================================================
    # Test 1: Token-level gradient on clean sentence
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 1: Clean sentence - where does gradient concentrate?")
    print("=" * 70)

    combined = "The capital city of France is Paris."
    importance, bpe_tokens = compute_token_level_gradient(
        model, embed_layer, tokenizer, combined, question, answer, device
    )
    imp_norm = importance / importance.sum()
    print(f"\n  Context: {combined}")
    print(f"  Question: {question} -> {answer}\n")
    for j, (tok, imp) in enumerate(zip(bpe_tokens, imp_norm)):
        bar = "█" * int(imp.item() * 80)
        print(f"  {j:3d} {imp.item():.4f} {bar:30s} '{tok}'")

    # ============================================================
    # Test 2: Answer followed by filler
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 2: Answer followed by increasing filler")
    print("=" * 70)

    fillers = [
        "",
        " This is nice.",
        " This is nice. The weather is good today.",
        " This is nice. The weather is good today. People enjoy walking in parks and gardens every morning.",
        " This is nice. The weather is good today. People enjoy walking in parks and gardens every morning. Technology continues to evolve and reshape how we communicate daily across the globe.",
    ]

    for filler in fillers:
        combined = f"The capital city of France is Paris.{filler}"
        importance, bpe_tokens = compute_token_level_gradient(
            model, embed_layer, tokenizer, combined, question, answer, device
        )

        # Find the BPE token for "Paris" 
        paris_imp = 0
        filler_imp = 0
        paris_found = False
        for j, tok in enumerate(bpe_tokens):
            if "paris" in tok.lower() or "Par" in tok:
                paris_imp += importance[j].item()
                paris_found = True
            elif paris_found:  # everything after Paris
                filler_imp += importance[j].item()

        total = importance.sum().item()
        print(f"\n  Filler length: {len(filler):3d} chars | Paris grad: {paris_imp/total:.3f} | Post-Paris grad: {filler_imp/total:.3f} | Ratio: {paris_imp/(filler_imp+1e-8):.2f}")

    # ============================================================
    # Test 3: Filler BEFORE vs AFTER the answer
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 3: Filler before vs after the answer")
    print("=" * 70)

    filler = "People enjoy walking in parks and gardens every morning. Technology continues to evolve rapidly."

    # Filler BEFORE answer
    combined_before = f"{filler} The capital city of France is Paris."
    imp_before, tokens_before = compute_token_level_gradient(
        model, embed_layer, tokenizer, combined_before, question, answer, device
    )
    imp_before_norm = imp_before / imp_before.sum()

    print(f"\n  FILLER BEFORE answer:")
    for j, (tok, imp) in enumerate(zip(tokens_before, imp_before_norm)):
        bar = "█" * int(imp.item() * 80)
        print(f"  {j:3d} {imp.item():.4f} {bar:30s} '{tok}'")

    # Filler AFTER answer
    combined_after = f"The capital city of France is Paris. {filler}"
    imp_after, tokens_after = compute_token_level_gradient(
        model, embed_layer, tokenizer, combined_after, question, answer, device
    )
    imp_after_norm = imp_after / imp_after.sum()

    print(f"\n  FILLER AFTER answer:")
    for j, (tok, imp) in enumerate(zip(tokens_after, imp_after_norm)):
        bar = "█" * int(imp.item() * 80)
        print(f"  {j:3d} {imp.item():.4f} {bar:30s} '{tok}'")

    # ============================================================
    # Test 4: Chunk-level with L=32 (larger chunks)
    # ============================================================
    print("\n" + "=" * 70)
    print("TEST 4: Larger chunks (L=32 words) - does answer chunk still win?")
    print("=" * 70)

    sentences = [
        "People enjoy walking in parks and gardens every morning when the sun rises. Technology continues to evolve and reshape how we communicate daily. Books provide knowledge and entertainment for readers of all ages worldwide.",
        "The capital city of France is Paris. This is a well known fact. Many tourists visit every year to see famous landmarks and cultural sites throughout the beautiful city.",
        "Ocean currents influence weather patterns across entire continents. Architecture reflects the cultural values and technological capabilities of different civilizations throughout recorded history.",
    ]
    combined = " ".join(sentences)
    tokens = simple_word_tokenize(combined)

    L = 32
    num_chunks = (len(tokens) + L - 1) // L

    importance, bpe_tokens = compute_token_level_gradient(
        model, embed_layer, tokenizer, combined, question, answer, device
    )

    # Map to chunks
    context_only_ids = tokenizer(combined, return_tensors="pt").to(device)
    context_bpe_len = context_only_ids.input_ids.shape[1]

    chunk_importance = torch.zeros(num_chunks)
    chunk_count = torch.zeros(num_chunks)
    for bpe_pos in range(min(context_bpe_len, importance.shape[0])):
        frac = bpe_pos / max(context_bpe_len, 1)
        chunk_idx = min(int(frac * num_chunks), num_chunks - 1)
        chunk_importance[chunk_idx] += importance[bpe_pos].item()
        chunk_count[chunk_idx] += 1

    valid = chunk_count > 0
    chunk_importance[valid] /= chunk_count[valid]
    chunk_norm = chunk_importance / chunk_importance.sum()

    print(f"\n  {len(tokens)} word tokens, {num_chunks} chunks of L={L}")
    for c in range(num_chunks):
        s, e = c * L, min((c + 1) * L, len(tokens))
        text = " ".join(tokens[s:e])
        has_answer = "paris" in text
        marker = " <<<" if has_answer else ""
        bar = "█" * int(chunk_norm[c].item() * 50)
        print(f"  Chunk {c}: {chunk_norm[c]:.3f} {bar:20s} {text[:80]}...{marker}")


if __name__ == "__main__":
    main()
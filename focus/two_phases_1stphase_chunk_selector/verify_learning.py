import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

def simple_word_tokenize(text: str):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())

def get_next_token_logprob(lm_model, lm_tokenizer, context_text: str, target_token: str, device: str):
    with torch.no_grad():
        inputs = lm_tokenizer(context_text, return_tensors="pt").to(device)
        outputs = lm_model(**inputs)
        logits = outputs.logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        target_ids = lm_tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_ids) == 0:
            return -100.0
        return log_probs[target_ids[0]].item()

device = "cuda"
model_name = "Qwen/Qwen2.5-3B"
print("Loading model...")
lm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
lm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
lm_model.eval()

sentences = [
    "In a trivia quiz, the capital of France is Paris, and many tourists visit Paris every year.",
    "Many museums and galleries throughout Paris display incredible art, while popular cafes in Paris serve delicious pastries.",
    "Paris is known for its romantic atmosphere, and lovers often stroll through Paris at sunset.",
]

query = "The capital city of France is"
target = " Paris"
L = 8

print("="*70)
print("REWARD ANALYSIS FOR ALL SENTENCES")
print("="*70)

for sent_idx, sentence in enumerate(sentences):
    tokens = simple_word_tokenize(sentence)
    print(f"\nSentence {sent_idx+1}: {sentence}")
    print(f"\nChunks:")
    for c in range(4):
        start = c * L
        end = min(start + L, len(tokens))
        chunk_toks = tokens[start:end]
        has_paris = "paris" in chunk_toks
        print(f"  Chunk {c}: {chunk_toks} {'← HAS PARIS' if has_paris else ''}")
    
    print(f"\nAll 2-chunk combinations ranked by reward:")
    results = []
    for i in range(4):
        for j in range(i+1, 4):
            chunk_i = tokens[i*L:min((i+1)*L, len(tokens))]
            chunk_j = tokens[j*L:min((j+1)*L, len(tokens))]
            context = " ".join(chunk_i + chunk_j) + " " + query
            logprob = get_next_token_logprob(lm_model, lm_tokenizer, context, target, device)
            has_paris_i = "paris" in chunk_i
            has_paris_j = "paris" in chunk_j
            has_paris = has_paris_i or has_paris_j
            results.append((i, j, logprob, has_paris))
    
    results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
    for rank, (i, j, logprob, has_paris) in enumerate(results_sorted, 1):
        marker = "✓ HAS PARIS" if has_paris else ""
        star = "★ BEST" if rank == 1 else ""
        print(f"  {rank}. Chunks [{i}, {j}]: {logprob:.3f}  {marker} {star}")
    
    # Check what the policy selected
    policy_selections = {
        0: [0, 1],  # From your output
        1: [1, 3],
        2: [2, 3]
    }
    selected = policy_selections[sent_idx]
    selected_result = [r for r in results if r[0] == min(selected) and r[1] == max(selected)][0]
    rank = results_sorted.index(selected_result) + 1
    print(f"\n  Policy selected [{selected[0]}, {selected[1]}]: rank {rank}/6")
    print(f"  {'✓ OPTIMAL' if rank == 1 else '✗ SUBOPTIMAL (could improve)'}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("The policy should learn to select chunks that:")
print("1. Contain 'Paris' (provides direct evidence)")
print("2. Achieve highest log probability for predicting ' Paris'")
print("\nIf policy selections are not rank 1, try:")
print("- More training steps")
print("- Higher learning rate")
print("- Different exploration strategy")

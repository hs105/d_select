import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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

combined_text = " ".join(sentences)
all_tokens = simple_word_tokenize(combined_text)

L = 8
query = "The capital city of France is"
target = " Paris"

print("Sentence 1 chunks (0-3):")
for c in range(4):
    start = c * L
    end = min(start + L, len(all_tokens))
    chunk_toks = all_tokens[start:end]
    has_paris = "paris" in chunk_toks
    marker = " ← HAS PARIS" if has_paris else ""
    print(f"  Chunk {c}: {chunk_toks}{marker}")

print("\n" + "="*70)
print("All 2-chunk combinations from Sentence 1 (chunks 0-3):")
print("="*70)

results = []
for i in range(4):
    for j in range(i+1, 4):
        chunk_i = all_tokens[i*L:min((i+1)*L, len(all_tokens))]
        chunk_j = all_tokens[j*L:min((j+1)*L, len(all_tokens))]
        
        context = " ".join(chunk_i + chunk_j) + " " + query
        logprob = get_next_token_logprob(lm_model, lm_tokenizer, context, target, device)
        
        has_paris_i = "paris" in chunk_i
        has_paris_j = "paris" in chunk_j
        has_paris = has_paris_i or has_paris_j
        
        results.append((i, j, logprob, has_paris))

# Sort by reward (best first)
results_sorted = sorted(results, key=lambda x: x[2], reverse=True)

for rank, (i, j, logprob, has_paris) in enumerate(results_sorted, 1):
    marker = "✓ HAS PARIS" if has_paris else ""
    star = "★ SELECTED BY POLICY" if [i, j] == [1, 2] else ""
    print(f"{rank}. Chunks [{i}, {j}]: {logprob:.4f}  {marker} {star}")

print("\n" + "="*70)
print("CONCLUSION:")
policy_selection = [1, 2]
policy_result = [r for r in results if r[0] == 1 and r[1] == 2][0]
rank = results_sorted.index(policy_result) + 1
print(f"Policy selected chunks {policy_selection}, which ranks #{rank}/6")

if rank == 1:
    print("✓ PERFECT! Policy found the optimal chunks")
elif rank <= 2:
    print("✓ EXCELLENT! Policy found near-optimal chunks")
else:
    print(f"✗ Suboptimal: Could improve to rank 1")

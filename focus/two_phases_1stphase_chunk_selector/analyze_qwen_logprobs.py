import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

"""
This shows 
 The tokenizer is case-sensitive! Look at the top predictions:

' Paris' (capitalized, ID unknown): log_prob = -0.8755
'paris' (lowercase, ID=1732): log_prob = -13.0859
"""

def simple_word_tokenize(text: str):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())

def get_next_token_logprob(lm_model, lm_tokenizer, context_text: str, target_token: str, device: str, verbose=False):
    with torch.no_grad():
        inputs = lm_tokenizer(context_text, return_tensors="pt").to(device)
        outputs = lm_model(**inputs)
        logits = outputs.logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Get top predictions
        if verbose:
            top_k = 10
            top_logprobs, top_indices = torch.topk(log_probs, top_k)
            print(f"\nContext: '{context_text}'")
            print(f"Top {top_k} predictions:")
            for i, (logprob, idx) in enumerate(zip(top_logprobs, top_indices)):
                token = lm_tokenizer.decode([idx])
                print(f"  {i+1}. '{token}' -> log_prob = {logprob.item():.4f}")
        
        target_ids = lm_tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_ids) == 0:
            target_ids = lm_tokenizer.encode(" " + target_token, add_special_tokens=False)
        
        if len(target_ids) == 0:
            print(f"Warning: Could not tokenize target '{target_token}'")
            return -100.0
        
        target_id = target_ids[0]
        target_logprob = log_probs[target_id].item()
        
        if verbose:
            print(f"\nTarget '{target_token}' (ID={target_id}): log_prob = {target_logprob:.4f}")
        
        return target_logprob

# Load model
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
print("Model loaded!\n")

# Test sentences
sentence1 = "In a trivia quiz, the capital of France is Paris, and many tourists visit Paris every year."
tokens1 = simple_word_tokenize(sentence1)

L = 8  # Chunk size
C = 4  # Number of chunks

print("="*70)
print("SENTENCE 1 ANALYSIS")
print("="*70)
print(f"Sentence: {sentence1}\n")

# Show chunks
print(f"Chunks (L={L}):")
for c in range(C):
    start = c * L
    end = min(start + L, len(tokens1))
    chunk_toks = tokens1[start:end]
    has_paris = "paris" in chunk_toks
    marker = " ← HAS PARIS" if has_paris else ""
    print(f"  Chunk {c}: {chunk_toks}{marker}")

print("\n" + "="*70)
print("TESTING DIFFERENT CHUNK COMBINATIONS")
print("="*70)

query = "The capital city of France is"
target = "paris"

# Test 1: No context (baseline)
print("\n1. No context (query only):")
logprob = get_next_token_logprob(lm_model, lm_tokenizer, query, target, device, verbose=True)

# Test 2: Chunks [0, 3] (what the policy selected)
print("\n2. Chunks [0, 3] (policy selection):")
chunk0 = tokens1[0:8]
chunk3 = tokens1[24:32]
context = " ".join(chunk0 + chunk3) + " " + query
logprob = get_next_token_logprob(lm_model, lm_tokenizer, context, target, device, verbose=True)

# Test 3: Chunks [1, 2] (contain "Paris")
print("\n3. Chunks [1, 2] (contain 'Paris'):")
chunk1 = tokens1[8:16]
chunk2 = tokens1[16:24]
context = " ".join(chunk1 + chunk2) + " " + query
logprob = get_next_token_logprob(lm_model, lm_tokenizer, context, target, device, verbose=True)

# Test 4: All possible 2-chunk combinations
print("\n" + "="*70)
print("ALL 2-CHUNK COMBINATIONS")
print("="*70)
results = []
for i in range(C):
    for j in range(i+1, C):
        chunk_i = tokens1[i*L:min((i+1)*L, len(tokens1))]
        chunk_j = tokens1[j*L:min((j+1)*L, len(tokens1))]
        context = " ".join(chunk_i + chunk_j) + " " + query
        logprob = get_next_token_logprob(lm_model, lm_tokenizer, context, target, device, verbose=False)
        has_paris_i = "paris" in chunk_i
        has_paris_j = "paris" in chunk_j
        has_paris = has_paris_i or has_paris_j
        results.append((i, j, logprob, has_paris))
        print(f"Chunks [{i}, {j}]: log_prob = {logprob:.4f}  {'✓ HAS PARIS' if has_paris else ''}")

print("\n" + "="*70)
print("SORTED BY LOG PROBABILITY (BEST TO WORST)")
print("="*70)
results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
for i, j, logprob, has_paris in results_sorted:
    print(f"Chunks [{i}, {j}]: log_prob = {logprob:.4f}  {'✓ HAS PARIS' if has_paris else ''}")

import json
import re

def simple_word_tokenize(text: str):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())

with open('/root/data/test_data.json', 'r') as f:
    test_data = json.load(f)

# Find the Paris example (idx 8)
paris_example = test_data[8]

print("="*70)
print("PARIS FAILURE - TEST EXAMPLE 8")
print("="*70)
print(f"\nQuestion: {paris_example['question']}")
print(f"Answer: {paris_example['answer']}")

print("\nSentences:")
for i, sent in enumerate(paris_example['sentences']):
    print(f"  {i}: {sent}")

combined = " ".join(paris_example['sentences'])
tokens = simple_word_tokenize(combined)

L = 8
print(f"\nAll chunks (L={L}):")
for c in range(12):
    start = c * L
    end = min(start + L, len(tokens))
    chunk_toks = tokens[start:end]
    
    has_paris = "paris" in chunk_toks
    
    marker = ""
    if c in [7, 8]:
        marker += " ← MODEL SELECTED"
    if c in [1, 2]:
        marker += " ← GROUND TRUTH"
    if has_paris:
        marker += " (contains 'paris')"
    
    print(f"  Chunk {c:2d}: {chunk_toks}{marker}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print("Model selected chunks 7 and 8")
print("These chunks likely don't contain 'paris' at all")
print("Or they contain irrelevant context")
print("\nGround truth chunks 1 and 2 contain the answer")
print("\nThis is a complete failure - model learned wrong pattern")

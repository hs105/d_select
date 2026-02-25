"""
Analyze why the model failed on certain examples
"""
import json
import re

def simple_word_tokenize(text: str):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())

# Load results
with open('/root/data/test_data.json', 'r') as f:
    test_data = json.load(f)

# Find the metrics file (get the most recent one)
import os
import glob
log_dir = '/root/data/logs'
metrics_files = glob.glob(os.path.join(log_dir, '*_metrics.json'))
latest_metrics = max(metrics_files, key=os.path.getmtime)

with open(latest_metrics, 'r') as f:
    metrics = json.load(f)

print("="*70)
print("FAILURE ANALYSIS")
print("="*70)

test_results = metrics['test_results']

# Analyze failures
failures = [r for r in test_results if r['status'] == 'failed']
successes = [r for r in test_results if r['status'] == 'perfect']

print(f"\nFailures: {len(failures)}/10")
print(f"Successes: {len(successes)}/10")

print("\n" + "="*70)
print("DETAILED FAILURE ANALYSIS")
print("="*70)

L = 8

for failure in failures:
    idx = failure['idx']
    example = test_data[idx]
    
    print(f"\n{'='*70}")
    print(f"Test {idx}: {failure['question']}")
    print(f"Answer: {failure['answer']}")
    print(f"{'='*70}")
    
    # Show all sentences
    print("\nSentences:")
    for i, sent in enumerate(example['sentences']):
        print(f"  {i}: {sent}")
    
    # Tokenize and show chunks
    combined = " ".join(example['sentences'])
    tokens = simple_word_tokenize(combined)
    
    print(f"\nChunks (L={L}):")
    for c in range(12):
        start = c * L
        end = min(start + L, len(tokens))
        chunk_toks = tokens[start:end]
        
        has_answer = example['answer_lower'] in chunk_toks
        is_selected = c in failure['selected_chunks']
        is_answer_chunk = c in failure['answer_chunks']
        
        markers = []
        if has_answer:
            markers.append("HAS ANSWER")
        if is_selected:
            markers.append("SELECTED")
        if is_answer_chunk:
            markers.append("GROUND TRUTH")
        
        marker_str = " ← " + ", ".join(markers) if markers else ""
        print(f"  Chunk {c:2d}: {chunk_toks}{marker_str}")
    
    print(f"\nWhat went wrong:")
    print(f"  Model selected chunks: {failure['selected_chunks']}")
    print(f"  Should have selected: {failure['answer_chunks']}")
    print(f"  Reward achieved: {failure['reward']:.3f}")

print("\n" + "="*70)
print("SUCCESS ANALYSIS")
print("="*70)

for success in successes[:3]:  # Show first 3 successes
    idx = success['idx']
    example = test_data[idx]
    
    print(f"\nTest {idx}: {success['question']}")
    print(f"  Selected: {success['selected_chunks']} = Ground truth: {success['answer_chunks']}")
    print(f"  Reward: {success['reward']:.3f}")

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)
print("1. Increase training steps (try 2000-3000)")
print("2. Lower learning rate (try 5e-4 or 1e-4)")
print("3. Add curriculum learning (easier examples first)")
print("4. Increase model capacity (larger d_model)")
print("5. Add regularization to prevent overfitting to noise")

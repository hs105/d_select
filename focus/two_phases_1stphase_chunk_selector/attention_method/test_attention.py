"""
Study: Does Qwen's attention reveal which chunks matter for prediction?

Feed full context + question, predict the answer, extract attention weights,
and see if they concentrate on the answer-bearing chunk.

This appears very bad. 

I did show both (average over all layers and only the last layer), and the last layer results were also bad (2/10, only hitting when answer was in chunk 0).
But the deeper issue is that even the last layer has strong positional bias. Look at the last layer attention for example 6 (Jupiter, GT=chunk 2):
Last layer: [0.140  0.309  0.089  0.023  0.177  0.059  0.202]
Chunk 1 (0.309) and chunk 6 (0.202) dominate, but chunk 1 is just "to make new discoveries about our world ." — generic filler. The answer chunk 2 (0.089) gets less attention than three irrelevant chunks.
The problem isn't which layer we look at — it's that attention weights ≠ importance. The model might read the key fact from chunk 2 in an earlier layer, store it in the residual stream, and by the last layer it no longer needs to attend there. 
The information has already been extracted and is flowing through the hidden states.
"""
import re
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np


def simple_word_tokenize(text):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Qwen...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",  # Need full attention weights, not flash attention
    )
    model.eval()
    
    # Load test data
    with open('/root/data/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    L = 8  # chunk size in our word tokenizer
    
    print("\n" + "=" * 70)
    print("ATTENTION ANALYSIS: Which chunks does Qwen attend to?")
    print("=" * 70)
    
    for i, ex in enumerate(test_data):
        combined = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined)
        
        # Find ground truth answer chunks
        answer_lower = ex["answer_lower"]
        gt_chunks = []
        for chunk_idx in range(len(tokens) // L + 1):
            start = chunk_idx * L
            end = min(start + L, len(tokens))
            chunk_toks = tokens[start:end]
            if answer_lower in chunk_toks:
                gt_chunks.append(chunk_idx)
        
        # Build the full input: context + question
        # We want to see what the model attends to when predicting the answer
        input_text = combined + " " + ex["question"]
        answer_text = " " + ex["answer"]
        full_text = input_text + answer_text
        
        # Tokenize
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
        
        context_len = inputs.input_ids.shape[1]
        full_len = full_inputs.input_ids.shape[1]
        
        if full_len <= context_len:
            print(f"  {i+1}: Skipping (no answer tokens)")
            continue
        
        # Forward pass with attention output
        with torch.no_grad():
            outputs = model(**full_inputs, output_attentions=True)
        
        # Get attention at the position just before the first answer token
        # This is the position that predicts the first answer token
        pred_pos = context_len - 1  # position that predicts token at context_len
        
        # Collect attention from all layers and heads at this position
        # attentions: tuple of (batch, heads, seq_len, seq_len) per layer
        num_layers = len(outputs.attentions)
        num_heads = outputs.attentions[0].shape[1]
        
        # Average attention across all layers and heads at the prediction position
        all_attn = []
        for layer_idx in range(num_layers):
            attn = outputs.attentions[layer_idx][0, :, pred_pos, :pred_pos+1]  # [heads, pred_pos+1]
            all_attn.append(attn)
        
        # [num_layers, num_heads, context_len]
        all_attn = torch.stack(all_attn)  # [layers, heads, context_len]
        
        # Average over layers and heads
        avg_attn = all_attn.mean(dim=(0, 1))  # [context_len]
        
        # Also look at last layer (often most task-relevant)
        last_layer_attn = all_attn[-1].mean(dim=0)  # [context_len]
        
        # Now map BPE token positions back to our word-level chunks
        # First, figure out which BPE tokens correspond to the context (before question)
        context_only = tokenizer(combined, return_tensors="pt").to(device)
        context_bpe_len = context_only.input_ids.shape[1]
        
        # Map each BPE position to a chunk index
        # We need to align BPE tokens to word-level chunks
        # Approximate: divide context BPE tokens proportionally into chunks
        num_word_tokens = len(tokens)
        num_chunks = (num_word_tokens + L - 1) // L
        
        # For each BPE token in context, assign to a chunk proportionally
        chunk_attn_avg = torch.zeros(num_chunks, device=device)
        chunk_attn_last = torch.zeros(num_chunks, device=device)
        chunk_count = torch.zeros(num_chunks, device=device)
        
        for bpe_pos in range(min(context_bpe_len, avg_attn.shape[0])):
            # Map BPE position to approximate chunk
            frac = bpe_pos / max(context_bpe_len, 1)
            chunk_idx = min(int(frac * num_chunks), num_chunks - 1)
            chunk_attn_avg[chunk_idx] += avg_attn[bpe_pos].item()
            chunk_attn_last[chunk_idx] += last_layer_attn[bpe_pos].item()
            chunk_count[chunk_idx] += 1
        
        # Normalize by count
        valid = chunk_count > 0
        chunk_attn_avg[valid] /= chunk_count[valid]
        chunk_attn_last[valid] /= chunk_count[valid]
        
        # Also compute attention on question tokens (everything after context)
        question_attn_avg = avg_attn[context_bpe_len:].sum().item() if context_bpe_len < avg_attn.shape[0] else 0
        
        # Find top-2 chunks by attention
        top2_avg = torch.topk(chunk_attn_avg[:num_chunks], k=min(2, num_chunks)).indices.tolist()
        top2_last = torch.topk(chunk_attn_last[:num_chunks], k=min(2, num_chunks)).indices.tolist()
        
        top2_avg_sorted = sorted(top2_avg)
        top2_last_sorted = sorted(top2_last)
        
        hit_avg = len(set(top2_avg) & set(gt_chunks)) > 0
        hit_last = len(set(top2_last) & set(gt_chunks)) > 0
        
        # Display
        status_avg = "✓" if hit_avg else "✗"
        status_last = "✓" if hit_last else "✗"
        
        # Normalize for display
        chunk_attn_avg_norm = chunk_attn_avg[:num_chunks] / chunk_attn_avg[:num_chunks].sum().clamp_min(1e-8)
        chunk_attn_last_norm = chunk_attn_last[:num_chunks] / chunk_attn_last[:num_chunks].sum().clamp_min(1e-8)
        
        avg_str = " ".join([f"{v:.3f}" for v in chunk_attn_avg_norm.tolist()])
        last_str = " ".join([f"{v:.3f}" for v in chunk_attn_last_norm.tolist()])
        
        print(f"\n  {i+1}. {ex['answer']:15s} | GT chunks={gt_chunks}")
        print(f"     Avg  all layers {status_avg}: top2={top2_avg_sorted} | Attn: [{avg_str}]")
        print(f"     Last layer      {status_last}: top2={top2_last_sorted} | Attn: [{last_str}]")
        
        # Show which tokens are in each chunk for interpretability
        for c in range(min(num_chunks, 12)):
            start = c * L
            end = min(start + L, len(tokens))
            chunk_text = " ".join(tokens[start:end])
            marker = " <<<" if c in gt_chunks else ""
            print(f"     Chunk {c}: {chunk_text}{marker}")


if __name__ == "__main__":
    main()
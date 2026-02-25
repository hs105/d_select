"""
Test: Can the pretrained LM identify which chunks contain the answer?
Feed it all chunks + question, ask it to pick the most relevant ones.
"""
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def simple_word_tokenize(text):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())


def chunk_text(tokens, L=8):
    chunks = []
    for i in range(0, len(tokens), L):
        chunk = tokens[i:i+L]
        if any(t.isalpha() for t in chunk):  # skip pure padding
            chunks.append(" ".join(chunk))
    return chunks


def find_answer_chunks(tokens, answer_lower, L=8):
    answer_chunks = []
    for chunk_idx in range(len(tokens) // L + 1):
        start = chunk_idx * L
        end = min(start + L, len(tokens))
        chunk_toks = tokens[start:end]
        if answer_lower in chunk_toks:
            answer_chunks.append(chunk_idx)
    return answer_chunks


def test_lm_chunk_selection(model, tokenizer, data, device, L=8):
    print("=" * 60)
    print("Testing LM chunk identification")
    print("=" * 60)
    
    correct = 0
    total = len(data)
    
    for i, ex in enumerate(data):
        combined = " ".join(ex["sentences"])
        tokens = simple_word_tokenize(combined)
        chunks = chunk_text(tokens, L)
        gt_chunks = find_answer_chunks(tokens, ex["answer_lower"], L)
        
        # Build prompt: show all chunks, ask which are relevant
        chunk_list = "\n".join([f"Chunk {j}: {c}" for j, c in enumerate(chunks)])
        
        prompt = f"""Here are text chunks:
{chunk_list}

Question: {ex['question']}
Answer: {ex['answer']}

Which chunk number(s) contain information most relevant to answering this question? Reply with ONLY the chunk number(s), separated by commas."""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.0,
                do_sample=False,
            )
        
        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Parse chunk numbers from response
        predicted = []
        for num in re.findall(r'\d+', response):
            n = int(num)
            if n < len(chunks):
                predicted.append(n)
        
        overlap = len(set(predicted) & set(gt_chunks))
        hit = overlap == len(gt_chunks) and len(gt_chunks) > 0
        if hit:
            correct += 1
        
        status = "✓" if hit else "✗"
        print(f"  {i+1} {status}: {ex['answer']:15s} | LM picked={predicted}, GT={gt_chunks} | Response: '{response}'")
    
    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading Qwen...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B",
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    
    with open('/root/data/test_data.json', 'r') as f:
        test_data = json.load(f)
    
    test_lm_chunk_selection(model, tokenizer, test_data, device)


if __name__ == "__main__":
    main()
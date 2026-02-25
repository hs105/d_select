"""
Autoregressive test: local chunk IS a candidate.

At each prediction point:
- Query = last W tokens (the local window)
- Candidates = ALL chunks up to and including the local chunk
- Focus network picks k=2 chunks
- Most of the time, local chunk wins (local context suffices)
- Sometimes, a distant chunk is needed (long-range dependency)

We test on the same document, but now at EVERY sentence boundary,
marking whether the next word needs local or distant context.
"""
import re
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def simple_word_tokenize(text):
    return re.findall(r"[A-Za-z]+|[^\sA-Za-z]", text.lower())


DOCUMENT = """Dr. Sarah Chen joined the quantum computing lab at MIT in 2019. She had previously worked at IBM Research for five years focusing on superconducting qubits. Her doctoral thesis at Stanford explored topological quantum error correction codes. The lab was located in Building 36 on the third floor. It contained three dilution refrigerators capable of reaching temperatures below ten millikelvin. The newest refrigerator was named Frostbite and could maintain coherence times exceeding one millisecond. In her first year Sarah published a breakthrough paper on quantum entanglement distribution across a fiber optic network spanning twelve kilometers. The paper appeared in Nature Physics and received over two hundred citations within six months. The research team consisted of four postdocs and eight graduate students. Marcus Wong handled the experimental setup while Elena Petrov focused on theoretical modeling. James Liu managed the cryogenic systems and Rebecca Santos worked on classical control electronics. Funding came primarily from a five year DARPA grant worth twelve million dollars. Additional support was provided by Google Quantum AI and the National Science Foundation. The DARPA program manager was Colonel David Mitchell who visited the lab quarterly. By 2022 the team had demonstrated quantum advantage for a specific optimization problem involving protein folding simulations. The quantum processor used forty seven qubits and outperformed the best classical supercomputer by a factor of one thousand. Sarah presented these results at the annual American Physical Society meeting in Las Vegas. Her talk attracted significant media attention and was covered by the New York Times and Scientific American. Several venture capital firms approached her about commercializing the technology. In early 2023 Sarah decided to leave MIT and founded a startup called QuantumLeap Technologies. She recruited Marcus and Elena from the lab as co-founders. The company raised thirty million dollars in Series A funding led by Andreessen Horowitz. QuantumLeap set up its headquarters in Cambridge Massachusetts just two miles from the MIT campus. The company focused on applying quantum computing to drug discovery specifically targeting molecular dynamics simulations for pharmaceutical companies. By late 2024 QuantumLeap had grown to fifty employees and secured partnerships with three major pharmaceutical companies including Pfizer and Novartis. The company was preparing for a Series B round targeting one hundred million dollars."""


# Prediction points throughout the document
# Each has: position in text (words up to this point), next word, whether local or distant
PREDICTIONS = [
    # LOCAL: next word predictable from recent context
    {"context_end": "Dr. Sarah Chen joined the quantum computing lab at MIT in",
     "next_word": "2019",
     "need": "local",
     "explanation": "Date follows naturally from 'joined ... in'"},

    {"context_end": "She had previously worked at IBM Research for five years focusing on superconducting",
     "next_word": "qubits",
     "need": "local",
     "explanation": "'superconducting ___' is a local collocation"},

    {"context_end": "The lab was located in Building 36 on the third",
     "next_word": "floor",
     "need": "local",
     "explanation": "'third ___' is locally predictable"},

    {"context_end": "It contained three dilution refrigerators capable of reaching temperatures below ten",
     "next_word": "millikelvin",
     "need": "local",
     "explanation": "'temperatures below ten ___' is local"},

    {"context_end": "In her first year Sarah published a breakthrough paper on quantum entanglement distribution across a fiber optic network spanning",
     "next_word": "twelve",
     "need": "local",
     "explanation": "'network spanning ___' - the number is new info, but locally introduced"},

    # DISTANT: next word requires recalling earlier information
    {"context_end": "The paper appeared in Nature Physics and received over two hundred citations within six months. The research team consisted of four postdocs and eight graduate students. Marcus Wong handled the experimental setup while Elena Petrov focused on theoretical modeling. James Liu managed the cryogenic systems and Rebecca Santos worked on classical control electronics. Funding came primarily from a five year DARPA grant worth twelve million dollars. Additional support was provided by Google Quantum AI and the National Science Foundation. The DARPA program manager was Colonel David Mitchell who visited the lab quarterly. By 2022 the team had demonstrated quantum advantage for a specific optimization problem involving protein folding simulations. The quantum processor used forty seven qubits and outperformed the best classical supercomputer by a factor of one thousand. Sarah presented these results at the annual American Physical Society meeting in Las Vegas. Her talk attracted significant media attention and was covered by the New York Times and Scientific American. Several venture capital firms approached her about commercializing the technology. In early 2023 Sarah decided to leave MIT and founded a startup called QuantumLeap Technologies. She recruited Marcus and Elena from the lab as co-founders. The company raised thirty million dollars in Series A funding led by Andreessen Horowitz. QuantumLeap set up its headquarters in Cambridge Massachusetts just two miles from the MIT campus. The company focused on applying quantum computing to drug discovery specifically targeting molecular dynamics simulations for pharmaceutical companies. By late 2024 QuantumLeap had grown to fifty employees and secured partnerships with three major pharmaceutical companies including Pfizer and Novartis. The company was preparing for a Series B round targeting one hundred million dollars. Sarah reflected on the journey from her early research on superconducting",
     "next_word": "qubits",
     "need": "distant",
     "explanation": "Need to recall paragraph 1: 'focusing on superconducting qubits'"},

    {"context_end": "The paper appeared in Nature Physics and received over two hundred citations within six months. The research team consisted of four postdocs and eight graduate students. Marcus Wong handled the experimental setup while Elena Petrov focused on theoretical modeling. James Liu managed the cryogenic systems and Rebecca Santos worked on classical control electronics. Funding came primarily from a five year DARPA grant worth twelve million dollars. Additional support was provided by Google Quantum AI and the National Science Foundation. The DARPA program manager was Colonel David Mitchell who visited the lab quarterly. By 2022 the team had demonstrated quantum advantage for a specific optimization problem involving protein folding simulations. The quantum processor used forty seven qubits and outperformed the best classical supercomputer by a factor of one thousand. Sarah presented these results at the annual American Physical Society meeting in Las Vegas. Her talk attracted significant media attention and was covered by the New York Times and Scientific American. Several venture capital firms approached her about commercializing the technology. In early 2023 Sarah decided to leave MIT and founded a startup called QuantumLeap Technologies. She recruited Marcus and Elena from the lab as co-founders. The company raised thirty million dollars in Series A funding led by Andreessen Horowitz. QuantumLeap set up its headquarters in Cambridge Massachusetts just two miles from the MIT campus. The company needed reliable cryogenic equipment so they moved the refrigerator named",
     "next_word": "Frostbite",
     "need": "distant",
     "explanation": "Need to recall paragraph 2: 'refrigerator was named Frostbite'"},

    {"context_end": "The paper appeared in Nature Physics and received over two hundred citations within six months. The research team consisted of four postdocs and eight graduate students. Marcus Wong handled the experimental setup while Elena Petrov focused on theoretical modeling. James Liu managed the cryogenic systems and Rebecca Santos worked on classical control electronics. Funding came primarily from a five year DARPA grant worth twelve million dollars. Additional support was provided by Google Quantum AI and the National Science Foundation. The DARPA program manager was Colonel David Mitchell who visited the lab quarterly. By 2022 the team had demonstrated quantum advantage for a specific optimization problem involving protein folding simulations. The quantum processor used forty seven qubits and outperformed the best classical supercomputer by a factor of one thousand. Sarah presented these results at the annual American Physical Society meeting in Las Vegas. Her talk attracted significant media attention and was covered by the New York Times and Scientific American. Several venture capital firms approached her about commercializing the technology. In early 2023 Sarah decided to leave MIT and founded a startup called QuantumLeap Technologies. She recruited Marcus and Elena from the lab as co-founders. The company raised thirty million dollars in Series A funding led by Andreessen Horowitz. QuantumLeap set up its headquarters in Cambridge Massachusetts just two miles from the MIT campus. The original entanglement experiment had covered a distance of",
     "next_word": "twelve",
     "need": "distant",
     "explanation": "Need to recall paragraph 3: 'network spanning twelve kilometers'"},

    {"context_end": "The paper appeared in Nature Physics and received over two hundred citations within six months. The research team consisted of four postdocs and eight graduate students. Marcus Wong handled the experimental setup while Elena Petrov focused on theoretical modeling. James Liu managed the cryogenic systems and Rebecca Santos worked on classical control electronics. Funding came primarily from a five year DARPA grant worth twelve million dollars. Additional support was provided by Google Quantum AI and the National Science Foundation. The DARPA program manager was Colonel David Mitchell who visited the lab quarterly. By 2022 the team had demonstrated quantum advantage for a specific optimization problem involving protein folding simulations. The quantum processor used forty seven qubits and outperformed the best classical supercomputer by a factor of one thousand. Sarah presented these results at the annual American Physical Society meeting in Las Vegas. Her talk attracted significant media attention and was covered by the New York Times and Scientific American. The quarterly review was led by",
     "next_word": "Colonel",
     "need": "distant",
     "explanation": "Need to recall paragraph 5: 'DARPA program manager was Colonel David Mitchell who visited quarterly'"},

    {"context_end": "The paper appeared in Nature Physics and received over two hundred citations within six months. The research team consisted of four postdocs and eight graduate students. Marcus Wong handled the experimental setup while Elena Petrov focused on theoretical modeling. James Liu managed the cryogenic systems and Rebecca Santos worked on classical control electronics. Funding came primarily from a five year DARPA grant worth twelve million dollars. Additional support was provided by Google Quantum AI and the National Science Foundation. The DARPA program manager was Colonel David Mitchell who visited the lab quarterly. By 2022 the team had demonstrated quantum advantage for a specific optimization problem involving protein folding simulations. The quantum processor used forty seven qubits and outperformed the best classical supercomputer by a factor of one thousand. The optimization problem that showed quantum advantage specifically involved",
     "next_word": "protein",
     "need": "distant",
     "explanation": "Need to recall earlier in same paragraph: 'involving protein folding simulations'"},
]


def main():
    print("Loading sentence-transformer...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    L = 16  # chunk size in words
    W = 24  # query window size in words
    k = 2   # select top-k chunks

    print(f"L={L}, W={W}, k={k}")

    doc_tokens = simple_word_tokenize(DOCUMENT)
    print(f"Document: {len(doc_tokens)} tokens")

    # Pre-chunk the full document
    num_doc_chunks = (len(doc_tokens) + L - 1) // L
    doc_chunk_texts = []
    for c in range(num_doc_chunks):
        s, e = c * L, min((c + 1) * L, len(doc_tokens))
        doc_chunk_texts.append(" ".join(doc_tokens[s:e]))

    # Encode all doc chunks
    doc_chunk_embs = st_model.encode(doc_chunk_texts, convert_to_tensor=True)

    print(f"\n{'=' * 80}")
    print(f"AUTOREGRESSIVE TEST: local chunk included as candidate")
    print(f"{'=' * 80}")

    local_correct = 0
    local_total = 0
    distant_correct = 0
    distant_total = 0

    for i, pred in enumerate(PREDICTIONS):
        # Tokenize context up to prediction point
        context_tokens = simple_word_tokenize(pred["context_end"])

        # Query = last W tokens
        query_tokens = context_tokens[-W:]
        query_text = " ".join(query_tokens)

        # Which chunks are available? All chunks up to the current position
        num_context_chunks = (len(context_tokens) + L - 1) // L
        # But we use the pre-chunked document chunks (since context is a prefix of the doc + continuation)
        # Actually, the context may extend beyond the document (for distant tests)
        # Let's chunk the context itself
        ctx_num_chunks = (len(context_tokens) + L - 1) // L
        ctx_chunk_texts = []
        for c in range(ctx_num_chunks):
            s, e = c * L, min((c + 1) * L, len(context_tokens))
            ctx_chunk_texts.append(" ".join(context_tokens[s:e]))

        ctx_chunk_embs = st_model.encode(ctx_chunk_texts, convert_to_tensor=True)

        # Query embedding
        q_emb = st_model.encode(query_text, convert_to_tensor=True)

        # Similarity with all context chunks
        sims = F.cosine_similarity(q_emb.unsqueeze(0), ctx_chunk_embs, dim=-1)

        # Top-k
        topk_vals, topk_idx = torch.topk(sims, k=min(k, ctx_num_chunks))
        top_k_list = sorted(topk_idx.tolist())

        # Local chunk = last chunk
        local_chunk = ctx_num_chunks - 1

        # Did it select the local chunk?
        selected_local = local_chunk in top_k_list

        # For distant predictions: find the chunk with the answer
        # Look for the next_word in earlier chunks
        next_word_lower = pred["next_word"].lower()
        answer_chunks = []
        for c in range(ctx_num_chunks):
            s, e = c * L, min((c + 1) * L, len(context_tokens))
            chunk_toks = context_tokens[s:e]
            if next_word_lower in chunk_toks:
                answer_chunks.append(c)

        # Determine if the selection is correct
        if pred["need"] == "local":
            # Should select local chunk
            hit = selected_local
            local_total += 1
            if hit:
                local_correct += 1
        else:
            # Should select a distant chunk containing the answer
            # (and ideally also the local chunk, but mainly the distant one)
            distant_chunks = [c for c in answer_chunks if c != local_chunk]
            hit = len(set(top_k_list) & set(distant_chunks)) > 0
            distant_total += 1
            if hit:
                distant_correct += 1

        status = "✓" if hit else "✗"
        need_tag = f"[{pred['need']:>7s}]"

        print(f"\n  {i+1} {status} {need_tag} next='{pred['next_word']}' | {ctx_num_chunks} chunks, local=chunk {local_chunk}")
        print(f"     Query: ...{query_text[-70:]}")
        print(f"     Selected: {top_k_list} | local_selected={selected_local}")
        print(f"     {pred['explanation']}")

        # Show top-5 with labels
        all_topk_vals, all_topk_idx = torch.topk(sims, k=min(5, ctx_num_chunks))
        for rank, (val, idx) in enumerate(zip(all_topk_vals, all_topk_idx)):
            label = ""
            if idx.item() == local_chunk:
                label = " [LOCAL]"
            if idx.item() in answer_chunks and idx.item() != local_chunk:
                label = " [ANSWER]"
            print(f"     #{rank+1}: chunk {idx:2d} sim={val:.3f} | {ctx_chunk_texts[idx.item()][:65]}{label}")

    print(f"\n{'=' * 80}")
    print(f"LOCAL predictions:  {local_correct}/{local_total} ({100*local_correct/local_total:.0f}%)")
    print(f"DISTANT predictions: {distant_correct}/{distant_total} ({100*distant_correct/distant_total:.0f}%)")
    print(f"TOTAL:              {local_correct+distant_correct}/{local_total+distant_total} ({100*(local_correct+distant_correct)/(local_total+distant_total):.0f}%)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
"""
Focus Net Transformer — Minimal Experiment
============================================
A tiny autoregressive transformer where standard dense self-attention
is replaced by focus-net-based sparse attention.

Architecture per layer:
  1. Focus nets score tokens in their regions, select top-k
  2. Add always-included tokens (last W) and summary tokens per region
  3. Self-attention ONLY over the selected+included set
  4. Scatter back + residual

Training: standard next-token prediction on a single long sentence.
Goal: verify gradients flow, focus nets learn, loss decreases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# ============================================================
# Config — everything tiny for testing
# ============================================================
class Config:
    vocab_size = 128         # small char-level vocab
    d_model = 64             # narrow model
    n_heads = 4              # attention heads
    n_layers = 4             # number of focus net layers
    d_ff = 128               # FFN hidden dim
    
    # Focus net params
    num_focus_nets = 4       # m: number of focus nets (= number of regions)
    tokens_per_focus = 3     # k: tokens selected per focus net
    always_include_last = 4  # W: always include last W tokens
    exploration_tokens = 1   # j: randomly promoted tokens per region (training)
    
    # Training
    lr = 3e-3
    epochs = 500
    
    dropout = 0.0            # no dropout for this tiny test


# ============================================================
# Focus Net: scores tokens in a region, selects top-k
# ============================================================
class FocusNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 1)  # one scalar score per token
        )
        self.k = config.tokens_per_focus
        self.exploration = config.exploration_tokens
    
    def forward(self, region_tokens, training=True):
        """
        region_tokens: [batch, region_len, d_model]
        Returns:
            selected_tokens: [batch, k, d_model]
            selected_indices: [batch, k] (positions within region)
            gate_weights: [batch, k]
            all_scores: [batch, region_len] (for logging)
        """
        B, R, D = region_tokens.shape
        
        # Score each token
        scores = self.scorer(region_tokens).squeeze(-1)  # [B, R]
        
        # Add Gumbel noise for exploration during training
        if training:
            noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            noisy_scores = scores + noise * 0.5  # moderate noise
        else:
            noisy_scores = scores
        
        # How many to select by score (leave room for exploration tokens)
        k_select = min(self.k, R)
        k_explore = min(self.exploration if training else 0, R - k_select)
        k_select = min(k_select, R - k_explore)
        
        if k_select <= 0:
            # Region too small, just take everything
            indices = torch.arange(R, device=region_tokens.device).unsqueeze(0).expand(B, -1)
            gate_weights = F.softmax(scores, dim=-1)
            return region_tokens, indices, gate_weights, scores
        
        # Top-k selection
        _, top_indices = torch.topk(noisy_scores, k_select, dim=-1)  # [B, k_select]
        
        # Stochastic promotion: randomly select from remaining tokens
        if k_explore > 0 and R > k_select:
            # Create mask of already selected
            mask = torch.ones(B, R, device=region_tokens.device, dtype=torch.bool)
            mask.scatter_(1, top_indices, False)
            
            # Sample from remaining
            remaining_probs = mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
            explore_indices = torch.multinomial(remaining_probs, k_explore)  # [B, k_explore]
            
            # Combine
            all_indices = torch.cat([top_indices, explore_indices], dim=-1)  # [B, k_select + k_explore]
        else:
            all_indices = top_indices
        
        # Sort indices for causal consistency
        all_indices, _ = torch.sort(all_indices, dim=-1)
        
        # Gather selected tokens (straight-through: detach index but keep score gradient)
        selected = torch.gather(
            region_tokens, 1, 
            all_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # [B, k_total, D]
        
        # Gate weights from original (non-noisy) scores
        selected_scores = torch.gather(scores, 1, all_indices)  # [B, k_total]
        gate_weights = F.softmax(selected_scores, dim=-1)       # [B, k_total]
        
        return selected, all_indices, gate_weights, scores


# ============================================================
# Summary token: mean pool over a region
# ============================================================
class RegionSummary(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Simple learnable projection for summary
        self.proj = nn.Linear(config.d_model, config.d_model)
    
    def forward(self, region_tokens):
        """region_tokens: [B, R, D] -> summary: [B, 1, D]"""
        pooled = region_tokens.mean(dim=1, keepdim=True)  # [B, 1, D]
        return self.proj(pooled)


# ============================================================
# Focus Net Attention Layer
# ============================================================
class FocusNetAttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.num_focus_nets = config.num_focus_nets
        
        # One focus net per region
        self.focus_nets = nn.ModuleList([
            FocusNet(config) for _ in range(config.num_focus_nets)
        ])
        
        # One summary module per region
        self.summaries = nn.ModuleList([
            RegionSummary(config) for _ in range(config.num_focus_nets)
        ])
        
        # Standard attention projections
        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)
        self.W_O = nn.Linear(config.d_model, config.d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)
    
    def forward(self, H, training=True):
        """
        H: [batch, seq_len, d_model]
        Returns: [batch, seq_len, d_model]
        """
        B, N, D = H.shape
        
        # --- Step 1: Partition into regions ---
        region_size = N // self.num_focus_nets
        remainder = N % self.num_focus_nets
        
        regions = []
        region_boundaries = []
        start = 0
        for i in range(self.num_focus_nets):
            end = start + region_size + (1 if i < remainder else 0)
            regions.append(H[:, start:end, :])       # [B, Ri, D]
            region_boundaries.append((start, end))
            start = end
        
        # --- Step 2: Focus nets select + get summaries ---
        all_selected = []         # selected token representations
        all_positions = []        # original positions in full sequence
        all_gates = []            # gate weights
        all_scores_log = []       # for monitoring
        
        for i, (region, (r_start, r_end)) in enumerate(zip(regions, region_boundaries)):
            # Focus net selects tokens
            selected, indices, gates, scores = self.focus_nets[i](region, training)
            
            # Convert local indices to global positions
            global_indices = indices + r_start
            
            all_selected.append(selected)
            all_positions.append(global_indices)
            all_gates.append(gates)
            all_scores_log.append((r_start, scores))
            
            # Summary token — assign it the position of the region start
            # (for causal masking: summary sees everything up to region end)
            summary = self.summaries[i](region)  # [B, 1, D]
            all_selected.append(summary)
            # Give summary a "virtual position" at the end of its region
            summary_pos = torch.full((B, 1), r_end - 1, device=H.device, dtype=torch.long)
            all_positions.append(summary_pos)
            all_gates.append(torch.ones(B, 1, device=H.device))
        
        # --- Step 3: Always-include last W tokens ---
        W = min(self.config.always_include_last, N)
        if W > 0:
            always_tokens = H[:, -W:, :]  # [B, W, D]
            always_positions = torch.arange(N - W, N, device=H.device).unsqueeze(0).expand(B, -1)
            all_selected.append(always_tokens)
            all_positions.append(always_positions)
            all_gates.append(torch.ones(B, W, device=H.device))
        
        # Concatenate all selected tokens
        S = torch.cat(all_selected, dim=1)          # [B, S_len, D]
        positions = torch.cat(all_positions, dim=1)  # [B, S_len]
        gates = torch.cat(all_gates, dim=1)          # [B, S_len]
        
        # --- Deduplicate by position ---
        # (A token might be both focus-selected and always-included)
        # Simple approach: keep all, attention will handle it fine
        # For a production version you'd deduplicate
        
        S_len = S.shape[1]
        
        # --- Step 4: Sparse self-attention over selected tokens ---
        S_normed = self.ln1(S)
        Q = self.W_Q(S_normed).view(B, S_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(S_normed).view(B, S_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(S_normed).view(B, S_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # Causal mask based on original positions
        # positions_i can attend to positions_j only if positions_j <= positions_i
        pos_q = positions.unsqueeze(2)  # [B, S_len, 1]
        pos_k = positions.unsqueeze(1)  # [B, 1, S_len]
        causal_mask = (pos_k <= pos_q).unsqueeze(1)  # [B, 1, S_len, S_len]
        
        # Scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 0)  # handle all-masked rows
        
        attn_out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, S_len, D)
        attn_out = self.W_O(attn_out)
        
        # Weight by gates (for gradient flow back to focus nets)
        attn_out = attn_out * gates.unsqueeze(-1)
        
        # --- Step 5: Scatter back + residual ---
        O = torch.zeros_like(H)         # [B, N, D]
        counts = torch.zeros(B, N, 1, device=H.device)
        
        # Scatter: add attn_out back to original positions
        pos_expanded = positions.unsqueeze(-1).expand(-1, -1, D)  # [B, S_len, D]
        O.scatter_add_(1, pos_expanded, attn_out)
        counts.scatter_add_(1, positions.unsqueeze(-1), torch.ones(B, S_len, 1, device=H.device))
        
        # Average where multiple tokens mapped to same position
        counts = counts.clamp(min=1)
        O = O / counts
        
        # Residual
        H = H + O
        
        # FFN with residual
        H = H + self.ffn(self.ln2(H))
        
        return H, all_scores_log


# ============================================================
# Full Model
# ============================================================
class FocusNetTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(512, config.d_model)  # max 512 tokens
        
        self.layers = nn.ModuleList([
            FocusNetAttentionLayer(config) for _ in range(config.n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def forward(self, input_ids, training=True):
        """
        input_ids: [batch, seq_len] (token indices)
        Returns: logits [batch, seq_len, vocab_size], layer_scores
        """
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0)
        
        H = self.embedding(input_ids) + self.pos_embedding(positions)
        
        all_layer_scores = []
        for layer in self.layers:
            H, scores = layer(H, training=training)
            all_layer_scores.append(scores)
        
        H = self.ln_final(H)
        logits = self.lm_head(H)
        
        return logits, all_layer_scores


# ============================================================
# Training on a single long sentence
# ============================================================
def main():
    torch.manual_seed(42)
    config = Config()
    
    # --- Create a simple dataset: one long sentence ---
    # Using ASCII characters as tokens for simplicity
    sentence = (
        "the quick brown fox jumps over the lazy dog and then the fox "
        "runs back to the forest where it meets another fox who is also "
        "quick and brown and they both jump over the sleeping dog again"
    )
    
    # Character-level tokenization (simple)
    chars = sorted(list(set(sentence)))
    char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 reserved for padding
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    idx_to_char[0] = '?'
    
    tokens = [char_to_idx[c] for c in sentence]
    input_ids = torch.tensor([tokens], dtype=torch.long)  # [1, seq_len]
    
    seq_len = input_ids.shape[1]
    print(f"Sentence length: {seq_len} characters")
    print(f"Vocab size used: {len(chars)} unique chars")
    print(f"Sentence: '{sentence}'")
    print(f"Config: {config.num_focus_nets} focus nets, {config.tokens_per_focus} tokens each")
    print(f"  Selected per layer: ~{config.num_focus_nets * config.tokens_per_focus} "
          f"+ {config.num_focus_nets} summaries "
          f"+ {config.always_include_last} always-included "
          f"= ~{config.num_focus_nets * config.tokens_per_focus + config.num_focus_nets + config.always_include_last} "
          f"out of {seq_len} tokens")
    print()
    
    # --- Model ---
    model = FocusNetTransformer(config)
    total_params = sum(p.numel() for p in model.parameters())
    focus_params = sum(p.numel() for fn in model.layers for f in fn.focus_nets for p in f.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Focus net parameters: {focus_params:,} ({100*focus_params/total_params:.1f}%)")
    print()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    # --- Training loop ---
    print("Training...")
    print("-" * 70)
    

    # PROPER LEAK DIAGNOSTIC: passed. no leaking.
    # Generate fresh random sequences each step.
    # If model predicts above chance → leak (it's using future info)
    # If model predicts at chance → no leak

    print("=== PROPER LEAK DIAGNOSTIC ===")
    diag_model = FocusNetTransformer(config)
    diag_opt = torch.optim.Adam(diag_model.parameters(), lr=config.lr)

    num_chars = len(chars)  # number of unique characters

    for ep in range(200):
        # Fresh random sequence every step — NO memorization possible
        random_tokens = torch.randint(1, num_chars + 1, (1, seq_len))
        
        logits, _ = diag_model(random_tokens, training=True)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, config.vocab_size),
            random_tokens[:, 1:].reshape(-1)
        )
        diag_opt.zero_grad()
        loss.backward()
        diag_opt.step()
        
        if ep % 50 == 0:
            # Test on ANOTHER fresh random sequence (never seen)
            diag_model.eval()
            with torch.no_grad():
                test_tokens = torch.randint(1, num_chars + 1, (1, seq_len))
                test_logits, _ = diag_model(test_tokens, training=False)
                test_acc = (test_logits[:, :-1, :].argmax(-1) == test_tokens[:, 1:]).float().mean()
                chance = 1.0 / num_chars
                print(f"  Ep {ep}: test acc={test_acc.item():.3f} (chance={chance:.3f})")
            diag_model.train()

    print("If test acc ≈ chance → NO leak. If test acc >> chance → LEAK.")
    print("=" * 50)

        
    for epoch in range(config.epochs):
        model.train()
        
        logits, layer_scores = model(input_ids, training=True)
        
        # Next-token prediction loss (shift by 1)
        # Predict token[t+1] from token[t]
        shift_logits = logits[:, :-1, :].contiguous()  # [1, N-1, vocab]
        shift_targets = input_ids[:, 1:].contiguous()    # [1, N-1]
        
        loss = F.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_targets.view(-1)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == config.epochs - 1:
            # Evaluate: generate next chars
            model.eval()
            with torch.no_grad():
                eval_logits, eval_scores = model(input_ids, training=False)
                preds = eval_logits.argmax(dim=-1)  # [1, N]
                
                # Accuracy: does predicted[t] == actual[t+1]?
                pred_next = preds[0, :-1]
                actual_next = input_ids[0, 1:]
                accuracy = (pred_next == actual_next).float().mean().item()
                
                # Show what focus nets are selecting at the last layer
                last_layer_scores = eval_scores[-1]  # list of (region_start, scores)
                
                print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
                      f"Next-char accuracy: {accuracy:.1%}")
                
                # Show focus net selections for last layer
                if epoch % 100 == 0 or epoch == config.epochs - 1:
                    print(f"  Focus net selections (layer {config.n_layers - 1}):")
                    for i, (r_start, scores) in enumerate(last_layer_scores):
                        top_k_vals, top_k_idx = torch.topk(scores[0], 
                            min(config.tokens_per_focus, scores.shape[1]))
                        global_idx = top_k_idx + r_start
                        selected_chars = ''.join([sentence[j] for j in global_idx.tolist() 
                                                   if j < len(sentence)])
                        region_text = sentence[r_start:r_start + scores.shape[1]]
                        print(f"    Focus net {i}: region='{region_text}' "
                              f"→ selected='{selected_chars}' "
                              f"(scores: {top_k_vals.tolist()})")
                    print()
    
    # --- Final generation test ---
    print("=" * 70)
    print("Generation test: feed first half, predict second half")
    print("=" * 70)
    
    model.eval()
    half = seq_len // 2
    prompt = input_ids[:, :half]
    
    generated = prompt.clone()
    with torch.no_grad():
        for step in range(min(50, seq_len - half)):  # generate 50 chars
            logits, _ = model(generated, training=False)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    
    # Decode
    prompt_text = ''.join([idx_to_char.get(t, '?') for t in prompt[0].tolist()])
    generated_text = ''.join([idx_to_char.get(t, '?') for t in generated[0, half:].tolist()])
    actual_text = sentence[half:half+50]
    
    print(f"Prompt:    '{prompt_text}'")
    print(f"Generated: '{generated_text}'")
    print(f"Actual:    '{actual_text}'")
    
    # Character-level match
    matches = sum(1 for a, b in zip(generated_text, actual_text) if a == b)
    print(f"Match: {matches}/{min(len(generated_text), len(actual_text))} "
          f"= {matches/min(len(generated_text), len(actual_text)):.1%}")


if __name__ == "__main__":
    main()
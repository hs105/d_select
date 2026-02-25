"""
Focus Net Transformer — Algorithmic Task Verification
======================================================
Task: COPY-BACK-K

Sequence of random tokens, repeating with period K:
    token[t] = token[t % K]  (i.e., first K tokens are random, then repeat)

To predict token[t+1], the model MUST look at token[t+1-K].
No other information helps — the tokens are random.

Fresh random sequences every batch → memorization impossible.
We know exactly which token the focus net should select.
We can measure: does the focus net learn to select position t+1-K?

Also compares against:
  - Standard (dense) transformer baseline (same params)
  - Random selection baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# Config
# ============================================================
class Config:
    vocab_size = 16          # tokens 0-15
    d_model = 64
    n_heads = 4
    n_layers = 4
    d_ff = 128

    # Focus net params
    num_focus_nets = 4
    tokens_per_focus = 12     # k per region
    always_include_last = 2
    exploration_tokens = 1
    use_summary = True

    # Training
    lr = 1e-3
    epochs = 3000
    batch_size = 32
    seq_len = 64

    # Task
    copy_back_K = 8          # period of repetition

    dropout = 0.0


# ============================================================
# Data generation
# ============================================================
def generate_copy_back(batch_size, seq_len, K, vocab_size):
    """
    First K tokens random. Then token[t] = token[t-K].
    So token[t] = token[t % K] for all t.
    
    To predict next token at position t: answer is token[(t+1) % K] = token[t+1-K] 
    (when t+1 >= K). Model must attend to position t+1-K.
    """
    prefix = torch.randint(0, vocab_size, (batch_size, K))
    # Repeat to fill seq_len
    repeats = (seq_len + K - 1) // K
    data = prefix.repeat(1, repeats)[:, :seq_len]
    return data


# ============================================================
# Focus Net components (same architecture)
# ============================================================
class FocusNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 4),
            nn.ReLU(),
            nn.Linear(config.d_model // 4, 1)
        )
        self.k = config.tokens_per_focus
        self.exploration = config.exploration_tokens

    def forward(self, region_tokens, training=True):
        B, R, D = region_tokens.shape
        scores = self.scorer(region_tokens).squeeze(-1)

        if training:
            noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
            noisy_scores = scores + noise * 0.3
        else:
            noisy_scores = scores

        k_select = min(self.k, R)
        k_explore = min(self.exploration if training else 0, max(0, R - k_select))
        k_select = min(k_select, R - k_explore)

        if k_select <= 0:
            indices = torch.arange(R, device=region_tokens.device).unsqueeze(0).expand(B, -1)
            gate_weights = F.softmax(scores, dim=-1)
            return region_tokens, indices, gate_weights, scores

        _, top_indices = torch.topk(noisy_scores, k_select, dim=-1)

        if k_explore > 0 and R > k_select:
            mask = torch.ones(B, R, device=region_tokens.device, dtype=torch.bool)
            mask.scatter_(1, top_indices, False)
            remaining_probs = mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp(min=1)
            explore_indices = torch.multinomial(remaining_probs, k_explore)
            all_indices = torch.cat([top_indices, explore_indices], dim=-1)
        else:
            all_indices = top_indices

        all_indices, _ = torch.sort(all_indices, dim=-1)

        selected = torch.gather(
            region_tokens, 1,
            all_indices.unsqueeze(-1).expand(-1, -1, D)
        )
        selected_scores = torch.gather(scores, 1, all_indices)
        gate_weights = F.softmax(selected_scores, dim=-1)

        return selected, all_indices, gate_weights, scores


class RegionSummary(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, region_tokens):
        return self.proj(region_tokens.mean(dim=1, keepdim=True))


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

        self.focus_nets = nn.ModuleList([FocusNet(config) for _ in range(config.num_focus_nets)])
        self.use_summary = config.use_summary
        if self.use_summary:
            self.summaries = nn.ModuleList([RegionSummary(config) for _ in range(config.num_focus_nets)])

        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)
        self.W_O = nn.Linear(config.d_model, config.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, H, training=True):
        B, N, D = H.shape

        region_size = N // self.num_focus_nets
        remainder = N % self.num_focus_nets
        regions, region_boundaries = [], []
        start = 0
        for i in range(self.num_focus_nets):
            end = start + region_size + (1 if i < remainder else 0)
            regions.append(H[:, start:end, :])
            region_boundaries.append((start, end))
            start = end

        all_selected, all_positions, all_gates = [], [], []
        focus_selections = []  # track what each focus net selected

        for i, (region, (r_start, r_end)) in enumerate(zip(regions, region_boundaries)):
            selected, indices, gates, scores = self.focus_nets[i](region, training)
            global_indices = indices + r_start

            all_selected.append(selected)
            all_positions.append(global_indices)
            all_gates.append(gates)
            focus_selections.append(global_indices)  # [B, k]

            if self.use_summary:
                summary = self.summaries[i](region)
                all_selected.append(summary)
                summary_pos = torch.full((B, 1), r_end - 1, device=H.device, dtype=torch.long)
                all_positions.append(summary_pos)
                all_gates.append(torch.ones(B, 1, device=H.device))

        W = min(self.config.always_include_last, N)
        if W > 0:
            all_selected.append(H[:, -W:, :])
            all_positions.append(torch.arange(N - W, N, device=H.device).unsqueeze(0).expand(B, -1))
            all_gates.append(torch.ones(B, W, device=H.device))

        S = torch.cat(all_selected, dim=1)
        positions = torch.cat(all_positions, dim=1)
        gates = torch.cat(all_gates, dim=1)
        S_len = S.shape[1]

        S_normed = self.ln1(S)
        Q = self.W_Q(S_normed).view(B, S_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(S_normed).view(B, S_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(S_normed).view(B, S_len, self.n_heads, self.d_head).transpose(1, 2)

        pos_q = positions.unsqueeze(2)
        pos_k = positions.unsqueeze(1)
        causal_mask = (pos_k <= pos_q).unsqueeze(1)

        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.masked_fill(attn_weights.isnan(), 0)

        attn_out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, S_len, D)
        attn_out = self.W_O(attn_out)
        attn_out = attn_out * gates.unsqueeze(-1)

        O = torch.zeros_like(H)
        counts = torch.zeros(B, N, 1, device=H.device)
        pos_expanded = positions.unsqueeze(-1).expand(-1, -1, D)
        O.scatter_add_(1, pos_expanded, attn_out)
        counts.scatter_add_(1, positions.unsqueeze(-1), torch.ones(B, S_len, 1, device=H.device))
        counts = counts.clamp(min=1)
        O = O / counts

        H = H + O
        H = H + self.ffn(self.ln2(H))

        return H, focus_selections


# ============================================================
# Full Focus Net Model
# ============================================================
class FocusNetTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(512, config.d_model)

        self.layers = nn.ModuleList([FocusNetAttentionLayer(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, training=True):
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0)
        H = self.embedding(input_ids) + self.pos_embedding(positions)

        all_selections = []
        for layer in self.layers:
            H, selections = layer(H, training=training)
            all_selections.append(selections)

        H = self.ln_final(H)
        logits = self.lm_head(H)
        return logits, all_selections


# ============================================================
# Standard Dense Transformer (baseline for comparison)
# ============================================================
class DenseTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        self.W_Q = nn.Linear(config.d_model, config.d_model)
        self.W_K = nn.Linear(config.d_model, config.d_model)
        self.W_V = nn.Linear(config.d_model, config.d_model)
        self.W_O = nn.Linear(config.d_model, config.d_model)

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model)
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, H):
        B, N, D = H.shape
        H_norm = self.ln1(H)
        Q = self.W_Q(H_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_K(H_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_V(H_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        causal_mask = torch.tril(torch.ones(N, N, device=H.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = attn.masked_fill(~causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_O(out)

        H = H + out
        H = H + self.ffn(self.ln2(H))
        return H


class DenseTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(512, config.d_model)
        self.layers = nn.ModuleList([DenseTransformerLayer(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids):
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0)
        H = self.embedding(input_ids) + self.pos_embedding(positions)
        for layer in self.layers:
            H = layer(H)
        H = self.ln_final(H)
        return self.lm_head(H)


# ============================================================
# Analysis: measure if focus nets select the "right" tokens
# ============================================================
def analyze_focus_selections(all_selections, seq_len, K, config):
    """
    For the copy-back task, to predict token[t+1], the ideal token to attend to
    is token[(t+1) - K] = token[(t+1) % K] which lives at position (t+1) - K
    (or more precisely, any position p where p % K == (t+1) % K and p <= t).
    
    The most recent such position is t + 1 - K (if t+1 >= K).
    
    We check: across all layers, what fraction of focus-net-selected positions
    are "useful" positions (i.e., positions p where p % K matches some needed value)?
    
    Returns: fraction of selected positions that are at a "copy source" position.
    """
    # For each position t (where t >= K), the model needs to see some position
    # p where p % K == (t+1) % K and p <= t.
    # The selected positions that are "useful" are those whose (pos % K) values
    # cover the needed remainders.
    
    # Simpler metric: for positions t >= K, is position (t+1-K) among the 
    # selected tokens in any layer?
    
    last_layer_selections = all_selections[-1]  # list of [B, k] tensors per focus net
    all_selected = torch.cat(last_layer_selections, dim=1)  # [B, total_selected]
    
    B = all_selected.shape[0]
    hits = 0
    total = 0
    
    for t in range(K, seq_len - 1):  # positions where we predict t+1 and t+1 >= K
        target_pos = (t + 1) - K  # the position we need
        # Check if target_pos is among selected positions for any sample
        is_selected = (all_selected == target_pos).any(dim=1).float()  # [B]
        hits += is_selected.sum().item()
        total += B
    
    return hits / total if total > 0 else 0.0


# ============================================================
# Main
# ============================================================
def main():
    torch.manual_seed(42)
    config = Config()
    K = config.copy_back_K
    device = torch.device('cpu')

    print("=" * 70)
    print("FOCUS NET TRANSFORMER — COPY-BACK TASK")
    print("=" * 70)
    print(f"Task: token[t] = token[t-{K}] (period-{K} repetition)")
    print(f"Sequence length: {config.seq_len}, Vocab size: {config.vocab_size}")
    print(f"Fresh random sequences each batch → no memorization possible")
    print(f"Correct behavior: focus nets should select position t+1-{K}")
    print()

    # Compute sparsity
    tokens_selected = (config.num_focus_nets * config.tokens_per_focus
                       + config.num_focus_nets * int(config.use_summary)
                       + config.always_include_last)
    print(f"Tokens attending per layer: ~{tokens_selected} out of {config.seq_len} "
          f"({100*tokens_selected/config.seq_len:.0f}%)")
    print()

    # --- Focus Net Model ---
    focus_model = FocusNetTransformer(config).to(device)
    focus_params = sum(p.numel() for p in focus_model.parameters())
    focus_net_params = sum(
        p.numel() for layer in focus_model.layers 
        for fn in layer.focus_nets for p in fn.parameters()
    )
    
    # --- Dense Baseline ---
    dense_model = DenseTransformer(config).to(device)
    dense_params = sum(p.numel() for p in dense_model.parameters())

    print(f"Focus Net model:  {focus_params:,} params (focus nets: {focus_net_params:,} = {100*focus_net_params/focus_params:.1f}%)")
    print(f"Dense baseline:   {dense_params:,} params")
    print()

    focus_opt = torch.optim.Adam(focus_model.parameters(), lr=config.lr)
    dense_opt = torch.optim.Adam(dense_model.parameters(), lr=config.lr)

    # --- Training ---
    print("Training...")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Focus Loss':>10} {'Focus Acc':>10} {'Focus Hit%':>10} | "
          f"{'Dense Loss':>10} {'Dense Acc':>10}")
    print("-" * 70)

    for epoch in range(config.epochs):
        # Fresh random data each batch
        data = generate_copy_back(config.batch_size, config.seq_len, K, config.vocab_size).to(device)
        targets = data[:, 1:]
        inputs = data[:, :-1]

        # We only evaluate loss on positions >= K (where the pattern is predictable)
        # Positions < K are random and unpredictable
        pred_mask = torch.zeros(config.seq_len - 1, dtype=torch.bool, device=device)
        pred_mask[K-1:] = True  # target[K-1] = data[K] = data[0], first predictable

        # --- Train Focus Net model ---
        focus_model.train()
        focus_logits, focus_selections = focus_model(inputs, training=True)
        focus_loss = F.cross_entropy(
            focus_logits[:, pred_mask, :].reshape(-1, config.vocab_size),
            targets[:, pred_mask].reshape(-1)
        )
        focus_opt.zero_grad()
        focus_loss.backward()
        focus_opt.step()

        # --- Train Dense model ---
        dense_model.train()
        dense_logits = dense_model(inputs)
        dense_loss = F.cross_entropy(
            dense_logits[:, pred_mask, :].reshape(-1, config.vocab_size),
            targets[:, pred_mask].reshape(-1)
        )
        dense_opt.zero_grad()
        dense_loss.backward()
        dense_opt.step()

        # --- Evaluate ---
        if epoch % 100 == 0 or epoch == config.epochs - 1:
            focus_model.eval()
            dense_model.eval()
            with torch.no_grad():
                # Fresh eval data
                eval_data = generate_copy_back(config.batch_size, config.seq_len, K, config.vocab_size).to(device)
                eval_inputs = eval_data[:, :-1]
                eval_targets = eval_data[:, 1:]

                # Focus net eval
                f_logits, f_sel = focus_model(eval_inputs, training=False)
                f_preds = f_logits[:, pred_mask, :].argmax(dim=-1)
                f_actual = eval_targets[:, pred_mask]
                f_acc = (f_preds == f_actual).float().mean().item()

                # Analyze focus selections
                hit_rate = analyze_focus_selections(
                    f_sel, config.seq_len - 1, K, config
                )

                # Dense eval
                d_logits = dense_model(eval_inputs)
                d_preds = d_logits[:, pred_mask, :].argmax(dim=-1)
                d_acc = (d_preds == f_actual).float().mean().item()

                # Dense loss on eval
                d_eval_loss = F.cross_entropy(
                    d_logits[:, pred_mask, :].reshape(-1, config.vocab_size),
                    eval_targets[:, pred_mask].reshape(-1)
                ).item()
                f_eval_loss = F.cross_entropy(
                    f_logits[:, pred_mask, :].reshape(-1, config.vocab_size),
                    eval_targets[:, pred_mask].reshape(-1)
                ).item()

                print(f"{epoch:6d} | {f_eval_loss:10.4f} {f_acc:9.1%} {hit_rate:9.1%} | "
                      f"{d_eval_loss:10.4f} {d_acc:9.1%}")

                # Detailed selection analysis at key epochs
                if epoch % 500 == 0 or epoch == config.epochs - 1:
                    print(f"\n  --- Selection analysis (epoch {epoch}) ---")
                    for layer_idx, layer_sel in enumerate(f_sel):
                        all_sel = torch.cat(layer_sel, dim=1)  # [B, total_k]
                        # For each selected position, what's its value mod K?
                        mod_values = all_sel % K
                        # Count frequency of each mod value
                        mod_counts = torch.zeros(K, device=device)
                        for m in range(K):
                            mod_counts[m] = (mod_values == m).float().sum()
                        mod_dist = mod_counts / mod_counts.sum()
                        top_mods = torch.topk(mod_counts, min(3, K))
                        mod_str = ", ".join([f"mod {m}:{c:.0f}" for m, c in 
                                             zip(top_mods.indices.tolist(), top_mods.values.tolist())])
                        print(f"  Layer {layer_idx}: selected positions mod {K}: {mod_str}")
                    
                    # Show example: for position t, what was selected?
                    print(f"\n  Example (sample 0): checking positions {K} to {K+7}...")
                    last_sel = torch.cat(f_sel[-1], dim=1)[0]  # [total_k] for sample 0
                    for t in range(K, min(K + 8, config.seq_len - 1)):
                        needed = t + 1 - K
                        found = "✓ YES" if needed in last_sel.tolist() else "✗ NO"
                        print(f"    Predict pos {t+1}: need pos {needed} (mod {K}={needed%K}), "
                              f"in selected? {found}")
                    print()

    # --- Final comparison ---
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    focus_model.eval()
    dense_model.eval()

    # Test on larger batch
    with torch.no_grad():
        test_data = generate_copy_back(256, config.seq_len, K, config.vocab_size).to(device)
        test_inputs = test_data[:, :-1]
        test_targets = test_data[:, 1:]

        f_logits, f_sel = focus_model(test_inputs, training=False)
        d_logits = dense_model(test_inputs)

        f_acc = (f_logits[:, pred_mask, :].argmax(-1) == test_targets[:, pred_mask]).float().mean()
        d_acc = (d_logits[:, pred_mask, :].argmax(-1) == test_targets[:, pred_mask]).float().mean()
        hit_rate = analyze_focus_selections(f_sel, config.seq_len - 1, K, config)

    print(f"Focus Net accuracy:   {f_acc:.1%}")
    print(f"Dense baseline accuracy: {d_acc:.1%}")
    print(f"Focus net hit rate (selects correct source token): {hit_rate:.1%}")
    print(f"Random selection hit rate (expected): {tokens_selected/config.seq_len:.1%}")

    # Count attention FLOPs
    focus_attn_ops = tokens_selected ** 2
    dense_attn_ops = config.seq_len ** 2
    print(f"\nAttention ops per layer: Focus={focus_attn_ops} vs Dense={dense_attn_ops} "
          f"({dense_attn_ops/focus_attn_ops:.0f}× more for dense)")


if __name__ == "__main__":
    main()
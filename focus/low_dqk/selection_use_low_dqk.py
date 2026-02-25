"""
Asymmetric Attention Dimension Experiment
==========================================
Test the hypothesis: selection (Q,K) needs fewer dimensions than value transfer (V).

Standard transformer: d_query = d_key = d_value = d_model
Our test:             d_query = d_key = d_select (sweep 2,4,8,16,32,64)
                      d_value = d_model (fixed at 64)

Task: copy-back-K (same as before). Fresh random data each batch.
Measure: accuracy vs d_select to find the minimum dimensionality for selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# Config
# ============================================================
class Config:
    vocab_size = 16
    d_model = 64
    n_heads = 4
    n_layers = 4
    d_ff = 128

    # Will be varied
    d_select = 64  # d_query = d_key

    lr = 1e-3
    epochs = 1500
    batch_size = 32
    seq_len = 64
    copy_back_K = 8

    dropout = 0.0


# ============================================================
# Data generation (same as before)
# ============================================================
def generate_copy_back(batch_size, seq_len, K, vocab_size):
    prefix = torch.randint(0, vocab_size, (batch_size, K))
    repeats = (seq_len + K - 1) // K
    data = prefix.repeat(1, repeats)[:, :seq_len]
    return data


# ============================================================
# Asymmetric Attention Layer: d_qk can differ from d_v
# ============================================================
class AsymmetricAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_select, d_ff):
        """
        d_select: dimension for Q and K (per head = d_select // n_heads)
        d_model:  dimension for V (per head = d_model // n_heads)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_select = d_select
        self.d_head_select = d_select // n_heads  # QK head dim
        self.d_head_value = d_model // n_heads     # V head dim

        # Q and K project to d_select (potentially small)
        self.W_Q = nn.Linear(d_model, d_select)
        self.W_K = nn.Linear(d_model, d_select)
        # V projects to full d_model (always large)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, H):
        B, N, D = H.shape
        H_norm = self.ln1(H)

        # Q, K in low-dimensional space
        Q = self.W_Q(H_norm).view(B, N, self.n_heads, self.d_head_select).transpose(1, 2)
        K = self.W_K(H_norm).view(B, N, self.n_heads, self.d_head_select).transpose(1, 2)
        # V in full-dimensional space
        V = self.W_V(H_norm).view(B, N, self.n_heads, self.d_head_value).transpose(1, 2)

        # Attention: selection happens in low-dim, value transfer in high-dim
        causal_mask = torch.tril(torch.ones(N, N, device=H.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head_select)
        attn = attn.masked_fill(~causal_mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V).transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_O(out)

        H = H + out
        H = H + self.ffn(self.ln2(H))
        return H


class AsymmetricTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(512, config.d_model)
        self.layers = nn.ModuleList([
            AsymmetricAttentionLayer(
                config.d_model, config.n_heads, config.d_select, config.d_ff
            ) for _ in range(config.n_layers)
        ])
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
# Training function
# ============================================================
def train_and_eval(config, d_select, device='cpu'):
    config.d_select = d_select

    # Ensure d_select is divisible by n_heads
    assert d_select % config.n_heads == 0, f"d_select={d_select} not divisible by n_heads={config.n_heads}"

    model = AsymmetricTransformer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    K = config.copy_back_K
    pred_mask = torch.zeros(config.seq_len - 1, dtype=torch.bool, device=device)
    pred_mask[K-1:] = True

    # Count parameters
    qk_params = sum(p.numel() for layer in model.layers for p in [layer.W_Q.weight, layer.W_Q.bias, layer.W_K.weight, layer.W_K.bias])
    v_params = sum(p.numel() for layer in model.layers for p in [layer.W_V.weight, layer.W_V.bias, layer.W_O.weight, layer.W_O.bias])
    total_params = sum(p.numel() for p in model.parameters())

    # Training
    best_acc = 0.0
    converge_epoch = None
    start_time = time.time()

    for epoch in range(config.epochs):
        model.train()
        data = generate_copy_back(config.batch_size, config.seq_len, K, config.vocab_size).to(device)
        inputs, targets = data[:, :-1], data[:, 1:]

        logits = model(inputs)
        loss = F.cross_entropy(
            logits[:, pred_mask, :].reshape(-1, config.vocab_size),
            targets[:, pred_mask].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Eval every 100 epochs
        if epoch % 100 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.no_grad():
                eval_data = generate_copy_back(config.batch_size * 4, config.seq_len, K, config.vocab_size).to(device)
                eval_inputs, eval_targets = eval_data[:, :-1], eval_data[:, 1:]
                eval_logits = model(eval_inputs)
                acc = (eval_logits[:, pred_mask, :].argmax(-1) == eval_targets[:, pred_mask]).float().mean().item()

                if acc > best_acc:
                    best_acc = acc
                if acc >= 0.99 and converge_epoch is None:
                    converge_epoch = epoch

    elapsed = time.time() - start_time

    return {
        'd_select': d_select,
        'best_acc': best_acc,
        'converge_epoch': converge_epoch,
        'total_params': total_params,
        'qk_params': qk_params,
        'v_params': v_params,
        'elapsed': elapsed,
    }


# ============================================================
# Main: sweep d_select
# ============================================================
def main():
    torch.manual_seed(42)
    config = Config()
    device = torch.device('cpu')

    print("=" * 80)
    print("ASYMMETRIC ATTENTION: d_query=d_key=d_select (swept) vs d_value=d_model (fixed)")
    print("=" * 80)
    print(f"Task: copy-back-{config.copy_back_K}, seq_len={config.seq_len}, vocab={config.vocab_size}")
    print(f"d_model={config.d_model}, n_heads={config.n_heads}, d_value_per_head={config.d_model//config.n_heads}")
    print(f"Training: {config.epochs} epochs, batch_size={config.batch_size}")
    print()

    d_select_values = [4, 8, 16, 32, 64]
    # Note: must be divisible by n_heads=4, so minimum is 4

    results = []

    for d_select in d_select_values:
        print(f"--- Training with d_select={d_select} (d_select_per_head={d_select//config.n_heads}) ---")
        torch.manual_seed(42)  # same init for fair comparison
        result = train_and_eval(config, d_select, device)
        results.append(result)
        converge_str = f"epoch {result['converge_epoch']}" if result['converge_epoch'] is not None else "did not converge"
        print(f"    Best accuracy: {result['best_acc']:.1%}, "
              f"Converged (≥99%): {converge_str}, "
              f"Time: {result['elapsed']:.1f}s")
        print(f"    QK params: {result['qk_params']:,}, V+O params: {result['v_params']:,}, "
              f"Total: {result['total_params']:,}")
        print()

    # Summary table
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'d_select':>8} {'d_sel/head':>10} {'Best Acc':>10} {'Converge':>12} "
          f"{'QK Params':>10} {'Total Params':>12} {'Time(s)':>8}")
    print("-" * 80)

    baseline_params = results[-1]['total_params']  # d_select=64 is baseline

    for r in results:
        converge_str = f"ep {r['converge_epoch']}" if r['converge_epoch'] is not None else "N/A"
        savings = (1 - r['total_params'] / baseline_params) * 100
        print(f"{r['d_select']:>8} {r['d_select']//config.n_heads:>10} {r['best_acc']:>9.1%} "
              f"{converge_str:>12} {r['qk_params']:>10,} {r['total_params']:>12,} "
              f"{r['elapsed']:>7.1f}")

    print()
    print("Key question: at what d_select does accuracy start to drop?")
    print("If accuracy holds at d_select << d_model, selection is low-dimensional.")

    # Compute attention FLOP comparison
    print()
    print("=" * 80)
    print("ATTENTION COMPUTE COMPARISON (per layer, per head)")
    print("=" * 80)
    n = config.seq_len
    d_v = config.d_model // config.n_heads
    for d_sel in d_select_values:
        d_s = d_sel // config.n_heads
        qk_flops = n * n * d_s          # Q @ K^T
        av_flops = n * n * d_v           # attn @ V
        total = qk_flops + av_flops
        baseline = n * n * d_v + n * n * d_v  # standard: both d_model/n_heads
        savings = (1 - total / baseline) * 100
        print(f"  d_select={d_sel:>2} (per_head={d_s:>2}): "
              f"QK={qk_flops:>8,} + AV={av_flops:>8,} = {total:>8,}  "
              f"({savings:>+5.1f}% vs standard)")


if __name__ == "__main__":
    main()
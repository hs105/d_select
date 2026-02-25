"""
Asymmetric Attention — Content-Based Selection Task
=====================================================
Task: KEY-VALUE RETRIEVAL

The sequence contains key-value pairs followed by a query:
    [k1 v1 k2 v2 k3 v3 ... kN vN SEP kQ]
    
The model must predict vQ — the value associated with key kQ.

Example with vocab=16, num_pairs=8:
    [3 7 | 9 2 | 5 11 | 1 4 | 6 0 | 8 13 | 12 15 | 10 14 | SEP | 5]
    Answer: 11 (because key 5 was paired with value 11)

Why this is hard for low d_select:
    - Selection is CONTENT-DEPENDENT: must find where key matches query
    - Selection is DYNAMIC: different queries need different keys
    - Position doesn't help: key-value pairs are in random order
    - The model must compute "does this key match my query?" — requires
      content comparison, not just positional rules

This directly tests: how many dimensions does content-based matching need?

We sweep d_select and compare against standard transformer.
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
    d_model = 64
    n_heads = 4
    n_layers = 4
    d_ff = 128

    # Task params
    num_pairs = 8          # number of key-value pairs
    num_keys = 16          # possible key values (0..num_keys-1)
    num_values = 16        # possible value values (0..num_values-1)
    # vocab: keys 0..15, values 16..31, SEP=32, total=33
    vocab_size = 33        # num_keys + num_values + 1 (SEP token)
    sep_token = 32

    # Sequence structure:
    # [k1, v1, k2, v2, ..., k_num_pairs, v_num_pairs, SEP, query_key]
    # seq_len = num_pairs * 2 + 2
    seq_len = 18           # 8 pairs × 2 + SEP + query = 18

    lr = 1e-3
    epochs = 3000
    batch_size = 64

    d_select = 64  # will be varied

    dropout = 0.0


# ============================================================
# Data generation
# ============================================================
def generate_kv_retrieval(batch_size, num_pairs, num_keys, num_values, sep_token):
    """
    Generate key-value retrieval sequences.
    
    Keys are unique within each sequence (no duplicate keys).
    Query is one of the existing keys.
    Target: the value paired with the query key.
    
    Sequence: [k1, v1, k2, v2, ..., kN, vN, SEP, query_key]
    Target at last position: corresponding value
    
    Keys are tokens 0..num_keys-1
    Values are tokens num_keys..num_keys+num_values-1
    SEP is token num_keys+num_values
    """
    B = batch_size
    
    sequences = torch.zeros(B, num_pairs * 2 + 2, dtype=torch.long)
    targets = torch.zeros(B, dtype=torch.long)
    
    for b in range(B):
        # Sample unique keys
        keys = torch.randperm(num_keys)[:num_pairs]
        # Sample random values (can repeat)
        values = torch.randint(0, num_values, (num_pairs,))
        
        # Place key-value pairs
        for i in range(num_pairs):
            sequences[b, 2 * i] = keys[i]                    # key token
            sequences[b, 2 * i + 1] = values[i] + num_keys   # value token (offset)
        
        # SEP token
        sequences[b, num_pairs * 2] = sep_token
        
        # Query: pick a random existing key
        query_idx = torch.randint(0, num_pairs, (1,)).item()
        sequences[b, num_pairs * 2 + 1] = keys[query_idx]
        
        # Target: the value paired with the query key
        targets[b] = values[query_idx] + num_keys  # offset to value token range
    
    return sequences, targets


# ============================================================
# Asymmetric Attention (same as before)
# ============================================================
class AsymmetricAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_select, d_ff):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head_select = d_select // n_heads
        self.d_head_value = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_select)
        self.W_K = nn.Linear(d_model, d_select)
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

        Q = self.W_Q(H_norm).view(B, N, self.n_heads, self.d_head_select).transpose(1, 2)
        K = self.W_K(H_norm).view(B, N, self.n_heads, self.d_head_select).transpose(1, 2)
        V = self.W_V(H_norm).view(B, N, self.n_heads, self.d_head_value).transpose(1, 2)

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
# Training
# ============================================================
def train_and_eval(config, d_select, device='cpu'):
    config.d_select = d_select
    assert d_select % config.n_heads == 0

    model = AsymmetricTransformer(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    qk_params = sum(p.numel() for layer in model.layers 
                     for p in [layer.W_Q.weight, layer.W_Q.bias, 
                               layer.W_K.weight, layer.W_K.bias])
    total_params = sum(p.numel() for p in model.parameters())

    last_pos = config.seq_len - 1  # position of query key → predict value

    best_acc = 0.0
    converge_epoch = None
    start_time = time.time()
    
    # Track progress for reporting
    progress = []

    for epoch in range(config.epochs):
        model.train()
        sequences, targets = generate_kv_retrieval(
            config.batch_size, config.num_pairs,
            config.num_keys, config.num_values, config.sep_token
        )
        sequences, targets = sequences.to(device), targets.to(device)

        logits = model(sequences)
        # Loss only on the last position (predicting the value for the query)
        loss = F.cross_entropy(logits[:, last_pos, :], targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == config.epochs - 1:
            model.eval()
            with torch.no_grad():
                eval_seq, eval_tgt = generate_kv_retrieval(
                    config.batch_size * 8, config.num_pairs,
                    config.num_keys, config.num_values, config.sep_token
                )
                eval_seq, eval_tgt = eval_seq.to(device), eval_tgt.to(device)
                eval_logits = model(eval_seq)
                eval_preds = eval_logits[:, last_pos, :].argmax(dim=-1)
                acc = (eval_preds == eval_tgt).float().mean().item()

                if acc > best_acc:
                    best_acc = acc
                if acc >= 0.95 and converge_epoch is None:
                    converge_epoch = epoch
                
                progress.append((epoch, acc, loss.item()))

    elapsed = time.time() - start_time

    return {
        'd_select': d_select,
        'best_acc': best_acc,
        'converge_epoch': converge_epoch,
        'total_params': total_params,
        'qk_params': qk_params,
        'elapsed': elapsed,
        'progress': progress,
    }


# ============================================================
# Main
# ============================================================
def main():
    torch.manual_seed(42)
    config = Config()
    device = torch.device('cpu')

    print("=" * 80)
    print("ASYMMETRIC ATTENTION — KEY-VALUE RETRIEVAL TASK")
    print("=" * 80)
    print(f"Task: sequence of {config.num_pairs} key-value pairs, then query a key")
    print(f"  Keys: 0..{config.num_keys-1}, Values: {config.num_keys}..{config.num_keys+config.num_values-1}, SEP: {config.sep_token}")
    print(f"  Sequence: [k1 v1 k2 v2 ... k{config.num_pairs} v{config.num_pairs} SEP query] → predict value")
    print(f"  Seq length: {config.seq_len}, Vocab: {config.vocab_size}")
    print(f"  Random chance: {1/config.num_values:.1%}")
    print(f"  Keys are unique per sequence, random order → position doesn't help")
    print(f"  Model MUST match query content against stored keys")
    print()
    print(f"Model: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}")
    print(f"Training: {config.epochs} epochs, batch_size={config.batch_size}")
    print()

    # Show example
    print("Example sequence:")
    seq, tgt = generate_kv_retrieval(1, config.num_pairs, config.num_keys, 
                                      config.num_values, config.sep_token)
    seq_list = seq[0].tolist()
    pairs_str = "  Pairs: "
    for i in range(config.num_pairs):
        k, v = seq_list[2*i], seq_list[2*i+1]
        pairs_str += f"({k}→{v-config.num_keys}) "
    print(pairs_str)
    print(f"  Query key: {seq_list[-1]}, Target value token: {tgt[0].item()} "
          f"(= value {tgt[0].item() - config.num_keys})")
    print()

    d_select_values = [4, 8, 16, 32, 64]
    results = []

    for d_select in d_select_values:
        print(f"--- d_select={d_select} (per_head={d_select//config.n_heads}) ---")
        torch.manual_seed(42)
        result = train_and_eval(config, d_select, device)
        results.append(result)

        converge_str = f"epoch {result['converge_epoch']}" if result['converge_epoch'] is not None else "did not converge"
        print(f"    Best accuracy: {result['best_acc']:.1%}, "
              f"Converged (≥95%): {converge_str}, "
              f"Time: {result['elapsed']:.1f}s")

        # Show training progress
        prog = result['progress']
        checkpoints = [p for p in prog if p[0] in [0, 200, 500, 1000, 1500, 2000, 2500, 2999]]
        prog_str = "    Progress: " + " → ".join([f"ep{e}:{a:.0%}" for e, a, _ in checkpoints])
        print(prog_str)
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY — KEY-VALUE RETRIEVAL (content-based selection)")
    print("=" * 80)
    print(f"{'d_select':>8} {'d/head':>6} {'Best Acc':>9} {'Converge':>14} "
          f"{'QK Params':>10} {'Total':>10} {'Time':>7}")
    print("-" * 80)
    
    baseline_params = results[-1]['total_params']
    for r in results:
        converge_str = f"ep {r['converge_epoch']}" if r['converge_epoch'] is not None else "not converged"
        param_save = (1 - r['total_params'] / baseline_params) * 100
        print(f"{r['d_select']:>8} {r['d_select']//config.n_heads:>6} {r['best_acc']:>8.1%} "
              f"{converge_str:>14} {r['qk_params']:>10,} {r['total_params']:>10,} "
              f"{r['elapsed']:>6.1f}s")

    print()
    print("COMPARISON WITH COPY-BACK TASK:")
    print("  Copy-back (positional selection): d_select=4 works → selection is trivially low-dim")
    print("  KV retrieval (content selection):  what d_select is needed?")
    print("  If d_select=16 or 32 works but d_select=4 fails → content matching needs")
    print("  more dimensions than positional selection, but still fewer than d_model")


if __name__ == "__main__":
    main()
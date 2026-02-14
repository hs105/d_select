"""
Asymmetric Attention Transformer
=================================
Standard causal transformer with decoupled Q/K and V dimensions.

d_select: dimension for Q and K (selection)
d_model:  dimension for V (value transfer) and everything else

When d_select = d_model, this is a standard transformer.
When d_select < d_model, selection is done in a lower-dimensional space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AsymmetricAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_select, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head_select = d_select // n_heads
        self.d_head_value = d_model // n_heads
        self.d_select = d_select

        # Q and K project to d_select (potentially small)
        self.W_Q = nn.Linear(d_model, d_select)
        self.W_K = nn.Linear(d_model, d_select)
        # V projects to full d_model
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, H, attn_mask=None):
        B, N, D = H.shape
        H_norm = self.ln1(H)

        Q = self.W_Q(H_norm).view(B, N, self.n_heads, self.d_head_select).transpose(1, 2)
        K = self.W_K(H_norm).view(B, N, self.n_heads, self.d_head_select).transpose(1, 2)
        V = self.W_V(H_norm).view(B, N, self.n_heads, self.d_head_value).transpose(1, 2)

        # Causal attention
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head_select)

        # Causal mask
        if attn_mask is None:
            attn_mask = torch.tril(
                torch.ones(N, N, device=H.device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, N, D)
        out = self.W_O(out)
        out = self.resid_dropout(out)

        H = H + out
        H = H + self.resid_dropout(self.ffn(self.ln2(H)))
        return H


class AsymmetricTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        d_select=None,       # None = d_model (standard transformer)
        max_seq_len=512,
        dropout=0.1,
        tie_weights=True,
    ):
        super().__init__()
        if d_select is None:
            d_select = d_model

        assert d_select % n_heads == 0, f"d_select={d_select} must be divisible by n_heads={n_heads}"
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model
        self.d_select = d_select
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            AsymmetricAttentionLayer(d_model, n_heads, d_select, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        """
        input_ids: [batch, seq_len]
        targets: [batch, seq_len] optional, for computing loss
        Returns: logits [batch, seq_len, vocab_size], loss (if targets given)
        """
        B, N = input_ids.shape
        assert N <= self.max_seq_len, f"Sequence length {N} exceeds max {self.max_seq_len}"

        positions = torch.arange(N, device=input_ids.device).unsqueeze(0)
        H = self.token_embedding(input_ids) + self.pos_embedding(positions)
        H = self.embed_dropout(H)

        for layer in self.layers:
            H = layer(H)

        H = self.ln_final(H)
        logits = self.lm_head(H)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        qk_params = sum(
            p.numel() for layer in self.layers
            for name, p in layer.named_parameters()
            if 'W_Q' in name or 'W_K' in name
        )
        v_params = sum(
            p.numel() for layer in self.layers
            for name, p in layer.named_parameters()
            if 'W_V' in name or 'W_O' in name
        )
        return {
            'total': total,
            'qk': qk_params,
            'vo': v_params,
            'other': total - qk_params - v_params,
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """Simple autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = input_ids[:, -self.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
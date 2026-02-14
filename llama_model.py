"""
LLaMA-Style Transformer with Asymmetric Attention
===================================================
Faithful to the LLaMA architecture:
  - RMSNorm (not LayerNorm)
  - SwiGLU FFN (not GELU)
  - Rotary Position Embeddings (RoPE, not learned)
  - No bias in linear layers
  - Pre-norm

The only modification: Q and K project to d_select (potentially < d_model),
while V projects to full d_model.

When d_select = d_model, this is a standard LLaMA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al., 2021)."""
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute cos/sin for max_seq_len
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim/2]
        self.register_buffer('cos_cached', freqs.cos())  # [seq_len, dim/2]
        self.register_buffer('sin_cached', freqs.sin())  # [seq_len, dim/2]

    def forward(self, seq_len):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary embeddings to x.
    x: [batch, n_heads, seq_len, head_dim]
    cos, sin: [seq_len, head_dim/2]
    """
    d = x.shape[-1]
    x1 = x[..., :d//2]
    x2 = x[..., d//2:]

    cos = cos[:x.shape[2], :d//2].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, d/2]
    sin = sin[:x.shape[2], :d//2].unsqueeze(0).unsqueeze(0)

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


class SwiGLU(nn.Module):
    """SwiGLU FFN (Shazeer, 2020). Used in LLaMA instead of GELU FFN."""
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)      # gate projection
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)       # down projection
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)       # up projection

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AsymmetricLlamaAttention(nn.Module):
    """
    LLaMA-style multi-head attention with asymmetric QK dimensions.

    Q, K project to d_select (can be < d_model).
    V projects to d_model.
    RoPE applied to Q and K.
    """
    def __init__(self, d_model, n_heads, d_select, max_seq_len=4096, bias=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head_qk = d_select // n_heads
        self.d_head_v = d_model // n_heads
        self.d_select = d_select

        # QK projections: potentially smaller
        self.W_Q = nn.Linear(d_model, d_select, bias=bias)
        self.W_K = nn.Linear(d_model, d_select, bias=bias)
        # V and O: full d_model
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

        # RoPE for QK heads
        self.rope = RotaryEmbedding(self.d_head_qk, max_seq_len=max_seq_len)

    def forward(self, H, attn_mask=None):
        B, N, D = H.shape

        Q = self.W_Q(H).view(B, N, self.n_heads, self.d_head_qk).transpose(1, 2)
        K = self.W_K(H).view(B, N, self.n_heads, self.d_head_qk).transpose(1, 2)
        V = self.W_V(H).view(B, N, self.n_heads, self.d_head_v).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(N)
        Q = apply_rotary_emb(Q, cos, sin)
        K = apply_rotary_emb(K, cos, sin)

        # Attention
        attn_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head_qk)

        # Causal mask
        if attn_mask is None:
            attn_mask = torch.tril(
                torch.ones(N, N, device=H.device, dtype=torch.bool)
            ).unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(~attn_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, N, D)
        return self.W_O(out)


class LlamaBlock(nn.Module):
    """Single LLaMA transformer block."""
    def __init__(self, d_model, n_heads, d_select, d_ff, max_seq_len=4096):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = AsymmetricLlamaAttention(d_model, n_heads, d_select, max_seq_len)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, H):
        H = H + self.attn(self.attn_norm(H))
        H = H + self.ffn(self.ffn_norm(H))
        return H


class AsymmetricLlama(nn.Module):
    """
    LLaMA-style causal language model with asymmetric attention.

    Config sizes (approximate):
      125M: d_model=768,  n_heads=12, n_layers=12, d_ff=2048
      350M: d_model=1024, n_heads=16, n_layers=24, d_ff=2816
    """
    def __init__(
        self,
        vocab_size,
        d_model=768,
        n_heads=12,
        n_layers=12,
        d_ff=2048,
        d_select=None,
        max_seq_len=1024,
        tie_weights=True,
    ):
        super().__init__()
        if d_select is None:
            d_select = d_model

        assert d_select % n_heads == 0, f"d_select={d_select} must divide by n_heads={n_heads}"
        assert d_model % n_heads == 0, f"d_model={d_model} must divide by n_heads={n_heads}"

        self.d_model = d_model
        self.d_select = d_select
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            LlamaBlock(d_model, n_heads, d_select, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])

        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

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
        B, N = input_ids.shape
        assert N <= self.max_seq_len, f"seq_len {N} > max {self.max_seq_len}"

        H = self.tok_emb(input_ids)

        for layer in self.layers:
            H = layer(H)

        H = self.norm(H)
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
            for name, p in layer.attn.named_parameters()
            if 'W_Q' in name or 'W_K' in name
        )
        vo_params = sum(
            p.numel() for layer in self.layers
            for name, p in layer.attn.named_parameters()
            if 'W_V' in name or 'W_O' in name
        )
        ffn_params = sum(
            p.numel() for layer in self.layers
            for name, p in layer.ffn.named_parameters()
        )
        return {
            'total': total,
            'qk': qk_params,
            'vo': vo_params,
            'ffn': ffn_params,
            'other': total - qk_params - vo_params - ffn_params,
        }

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
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
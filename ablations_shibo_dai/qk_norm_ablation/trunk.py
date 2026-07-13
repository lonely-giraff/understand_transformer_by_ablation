"""QK-Norm ablation: remove the per-head Q/K normalization from attention.

The reference GPT normalizes Q and K after RoPE and before the attention
dot-product::

    q, k = norm(q), norm(k)  # line 156 of core/model/gpt.py

This keeps attention logits bounded and training stable. Removing it is a
one-line architecture change; this trunk provides a ``NoQKNormAttention`` class
that drops that line, and a ``NoQKNormGPT`` that uses those blocks. Everything
else (RoPE, MLP, residual, LM head) is inherited from the reference GPT.

The only difference from the baseline is the QK-norm itself — same model
geometry, same data, same budget.
"""

import torch
import torch.nn as nn

from core.model.gpt import GPT, GPTConfig, Block, CausalSelfAttention, apply_rotary_emb, norm


class NoQKNormAttention(CausalSelfAttention):
    """Same attention, but WITHOUT per-head Q/K normalization (line 156 removed)."""

    def forward(self, x, cos_sin, kv_cache, block_mask=None):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings (position encoding)
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # NOTE: QK-norm REMOVED — the ONLY change from the reference Attention
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)

        # Attention: same four modes as the reference (FlexAttention / causal / cached / chunked)
        if block_mask is not None:
            if self.enable_gqa:
                n_rep = self.n_head // self.n_kv_head
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            from torch.nn.attention import flex_attention
            y = flex_attention(q, k, v, block_mask=block_mask)
        elif kv_cache is None or Tq == Tk:
            if self.enable_gqa:
                y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            if self.enable_gqa:
                y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
            else:
                y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            if self.enable_gqa:
                y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
            else:
                y = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class NoQKNormBlock(Block):
    """A transformer block whose attention removes QK-norm."""

    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        self.attn = NoQKNormAttention(config, layer_idx)
        from core.model.gpt import MLP
        self.mlp = MLP(config)


class NoQKNormGPT(GPT):
    """The reference GPT with QK-norm removed from every attention layer.
    Config-compatible with GPT (same GPTConfig), so it drives through the same orchestrator."""

    Config = GPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer.h = nn.ModuleList(
            [NoQKNormBlock(config, i) for i in range(config.n_layer)])

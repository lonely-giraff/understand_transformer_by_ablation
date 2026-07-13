"""RoPE → Learned Position Embedding ablation.

The reference GPT uses Rotary Position Embedding (RoPE) inside every attention
layer — position info is injected into Q and K via rotation. This trunk replaces
it with GPT-2-style learned absolute position embeddings (wpe): a trainable
lookup table added to token embeddings BEFORE the transformer blocks.

  * baseline   — reference GPT (RoPE in attention)
  * no_rope    — NoRoPEGPT: learned wpe embedding, NO RoPE in attention

Everything else (MLP, RMSNorm, residual, QK-norm, LM head) is inherited.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.gpt import GPT, GPTConfig, Block, CausalSelfAttention, apply_rotary_emb, norm


class NoRoPEAttention(CausalSelfAttention):
    """Same attention but WITHOUT RoPE — position info comes from wpe instead."""

    def forward(self, x, cos_sin, kv_cache, block_mask=None):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # RoPE REMOVED — position already encoded via wpe in the trunk
        q, k = norm(q), norm(k)  # QK-norm kept
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2)
        Tk = k.size(2)

        if block_mask is not None:
            if self.enable_gqa:
                n_rep = self.n_head // self.n_kv_head
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            from torch.nn.attention import flex_attention
            y = flex_attention(q, k, v, block_mask=block_mask)
        elif kv_cache is None or Tq == Tk:
            if self.enable_gqa:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif Tq == 1:
            if self.enable_gqa:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        else:
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device)
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            if self.enable_gqa:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class NoRoPEBlock(Block):
    """Transformer block with NoRoPEAttention (no rotary position encoding)."""

    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        self.attn = NoRoPEAttention(config, layer_idx)
        from core.model.gpt import MLP
        self.mlp = MLP(config)


class NoRoPEGPT(GPT):
    """GPT with learned absolute position embeddings (wpe) instead of RoPE.

    Position information is injected ONCE at the embedding level:
        x = wte(tokens) + wpe(positions)
    and NOT inside attention (no RoPE rotation). This is the GPT-2 scheme."""

    Config = GPTConfig

    def __init__(self, config):
        super().__init__(config)
        # Learned absolute position embedding (GPT-2 style)
        self.transformer.wpe = nn.Embedding(config.sequence_len, config.n_embd)
        # Replace blocks with NoRoPE versions
        self.transformer.h = nn.ModuleList(
            [NoRoPEBlock(config, i) for i in range(config.n_layer)])

    def init_weights(self):
        super().init_weights()
        # Standard init for the new wpe embedding
        self._init_weights(self.transformer.wpe)

    def forward(self, idx, token_types=None, kv_cache=None, block_mask=None):
        B, T = idx.size()

        # Token embeddings
        x = self.transformer.wte(idx)

        if token_types is not None:
            x = x + self.type_emb(token_types)

        # Add learned absolute position embeddings (GPT-2 style)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = x + self.transformer.wpe(pos)

        # Pass dummy cos_sin — blocks accept it for API compat, NoRoPEAttention ignores it
        cos_sin = (None, None)

        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache, block_mask)
        x = norm(x)
        return x

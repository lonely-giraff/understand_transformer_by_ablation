"""A GPT without RoPE — the architecture change this ablation studies.

Subclasses the reference GPT and skips rotary position embeddings in
attention. Everything else (attention, MLP, RMSNorm, QK-norm, the trunk
contract, the FLOPs estimate) is inherited. This is the whole architecture
edit — no fork of core.

The ablation: the baseline applies RoPE to Q and K before attention,
giving each token a position-dependent rotation. The ablated version skips
this — attention has NO positional information beyond causal masking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model.gpt import GPT, GPTConfig, Block, norm


class NoRoPEAttention(nn.Module):
    """CausalSelfAttention WITHOUT RoPE. Same weights/shape/init as the
    reference, but the RoPE line is removed from forward."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.enable_gqa = self.n_kv_head != self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache, block_mask=None):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # --- THE ONLY CHANGE: NO apply_rotary_emb ---
        # baseline:  cos, sin = cos_sin
        #            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # ablated:   q, k stay as-is — zero positional information in attention

        q, k = norm(q), norm(k)  # QK norm (kept, unrelated to RoPE)
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
            from torch.nn.attention.flex_attention import flex_attention
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
    """Block whose attention strips RoPE."""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.attn = NoRoPEAttention(config, layer_idx)


class NoRoPEGPT(GPT):
    """The reference GPT without RoPE. Config-compatible with GPT (same
    GPTConfig), so it drives through the same orchestrator as the baseline."""

    Config = GPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer.h = nn.ModuleList(
            [NoRoPEBlock(config, i) for i in range(config.n_layer)])

    def init_weights(self):
        super().init_weights()

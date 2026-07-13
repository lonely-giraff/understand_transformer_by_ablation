"""
variant_trunk.py — local GPT trunks for attn:mlp ratio ablation.

Three trunks, all with the SAME total parameters at each depth:
  - Standard:    d full blocks (attn + mlp), 33% attn params
  - AttnHeavyGPT: same total params, ~50-67% attn params
  - MLPHeavyGPT:  same total params, ~11-17% attn params

Block layout: extra-attn blocks first, then paired full blocks, extra-mlp last.
Self-contained: defines its own Block (with has_attn/has_mlp) so it doesn't
depend on a modified nanoinfra.
"""
import torch
import torch.nn as nn
from core.model.gpt import GPT, GPTConfig, CausalSelfAttention, MLP, norm


# Per-depth (N_attn, N_mlp) — constraint: 4*N_attn + 8*N_mlp = 12*d
TABLE = {
    "attn_heavy": {2: (4, 1), 3: (5, 2), 4: (8, 2), 6: (10, 4), 8: (12, 6)},
    "mlp_heavy":  {2: (2, 2), 3: (1, 4), 4: (2, 5), 6: (2, 8),  8: (4, 10)},
}


class VariantBlock(nn.Module):
    """A transformer block where attn and mlp can be independently toggled."""
    def __init__(self, config, layer_idx, has_attn=True, has_mlp=True):
        super().__init__()
        self.has_attn = has_attn
        self.has_mlp = has_mlp
        if has_attn:
            self.attn = CausalSelfAttention(config, layer_idx)
        if has_mlp:
            self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache, block_mask=None):
        if self.has_attn:
            x = x + self.attn(norm(x), cos_sin, kv_cache, block_mask)
        if self.has_mlp:
            x = x + self.mlp(norm(x))
        return x


def _build_blocks(config, n_attn, n_mlp):
    """[A]*extra_attn + [AM]*n_full + [M]*extra_mlp."""
    n_full = min(n_attn, n_mlp)
    extra_attn = n_attn - n_full
    extra_mlp = n_mlp - n_full
    blocks = []
    for _ in range(extra_attn):
        blocks.append(VariantBlock(config, len(blocks), has_attn=True, has_mlp=False))
    for _ in range(n_full):
        blocks.append(VariantBlock(config, len(blocks), has_attn=True, has_mlp=True))
    for _ in range(extra_mlp):
        blocks.append(VariantBlock(config, len(blocks), has_attn=False, has_mlp=True))
    return nn.ModuleList(blocks)


class _VariantGPT(GPT):
    """Base class: replaces standard blocks with a variant layout."""

    def __init__(self, config, n_attn, n_mlp):
        super().__init__(config)  # builds standard blocks — we swap below
        self.transformer["h"] = _build_blocks(config, n_attn, n_mlp)

    def init_weights(self):
        self.apply(self._init_weights)
        for block in self.transformer.h:
            if block.has_mlp:
                nn.init.zeros_(block.mlp.c_proj.weight)
            if block.has_attn:
                nn.init.zeros_(block.attn.c_proj.weight)
        nn.init.zeros_(self.type_emb.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            self.type_emb.to(dtype=torch.bfloat16)

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        n_attn_layers = sum(1 for b in self.transformer.h if b.has_attn)
        l, h = n_attn_layers, self.config.n_head
        q, t = self.config.n_embd // self.config.n_head, self.config.sequence_len
        return 6 * (nparams - nparams_embedding) + 12 * l * h * q * t


def _make_variant(name):
    """Create a trunk class for `name` ("attn_heavy" or "mlp_heavy")."""
    class VariantGPT(_VariantGPT):
        Config = GPTConfig
        def __init__(self, config):
            n_attn, n_mlp = TABLE[name][config.n_layer]
            super().__init__(config, n_attn, n_mlp)
    VariantGPT.__name__ = f"{name.title().replace('_','')}GPT"
    VariantGPT.__qualname__ = VariantGPT.__name__
    return VariantGPT


AttnHeavyGPT = _make_variant("attn_heavy")
MLPHeavyGPT = _make_variant("mlp_heavy")

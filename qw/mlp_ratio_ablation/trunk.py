"""MLP-ratio variants of the reference GPT — the architecture change this
ablation studies.

Subclasses the reference GPT and swaps in MLP blocks with a custom expansion
ratio. The standard transformer uses 4x::

    c_fc:  n_embd -> 4*n_embd
    c_proj: 4*n_embd -> n_embd

This module provides trunks that change that ratio — nothing else. Each ratio
is a separate class, so `model.trunk_class` selects it without any core edit.
"""

import torch.nn as nn

from core.model.gpt import GPT, GPTConfig, Block, CausalSelfAttention, norm

# ---------------------------------------------------------------------------
# Custom MLP with configurable ratio
# ---------------------------------------------------------------------------
class RatioMLP(nn.Module):
    """MLP whose hidden dimension = mlp_ratio * n_embd (standard = 4x)."""

    def __init__(self, config, mlp_ratio):
        super().__init__()
        hidden_dim = int(config.n_embd * mlp_ratio)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = nn.functional.relu(x).square()
        x = self.c_proj(x)
        return x


class RatioBlock(Block):
    """A transformer block whose MLP uses a custom expansion ratio."""

    def __init__(self, config, layer_idx, mlp_ratio):
        super().__init__(config, layer_idx)
        self.mlp = RatioMLP(config, mlp_ratio)


# ---------------------------------------------------------------------------
# Trunk classes — one per ratio (selected via model.trunk_class)
# ---------------------------------------------------------------------------
class MLPRatioGPT(GPT):
    """Base: rebuilds every block's MLP with a given ratio. Subclasses pin the value."""

    Config = GPTConfig
    mlp_ratio: float = 4.0

    def __init__(self, config):
        super().__init__(config)
        self.transformer.h = nn.ModuleList(
            [RatioBlock(config, i, self.mlp_ratio) for i in range(config.n_layer)])


class GPT_MLP2x(MLPRatioGPT):
    mlp_ratio = 2.0


class GPT_MLP4x(MLPRatioGPT):
    mlp_ratio = 4.0


class GPT_MLP6x(MLPRatioGPT):
    mlp_ratio = 6.0


class GPT_MLP8x(MLPRatioGPT):
    mlp_ratio = 8.0

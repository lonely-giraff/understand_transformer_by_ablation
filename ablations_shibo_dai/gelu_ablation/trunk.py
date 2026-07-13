"""ReLU² → GELU activation ablation: replace the MLP's activation function.

The reference GPT uses ReLU² (``F.relu(x).square()``) in its MLP — a modern
choice. This trunk swaps it for the classic GELU (``F.gelu(x)``), the standard
activation from the GPT-2 / BERT era.

Everything else (attention, RoPE, QK-norm, RMSNorm, residual, LM head) is
inherited from the reference GPT. The only difference is the activation function
in the feed-forward network.
"""

import torch.nn as nn
import torch.nn.functional as F

from core.model.gpt import GPT, GPTConfig, Block, MLP


class GELUMLP(MLP):
    """Same MLP geometry, but with GELU instead of ReLU²."""

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)           # ← classic GELU replaces ReLU²
        x = self.c_proj(x)
        return x


class GELUBlock(Block):
    """A transformer block whose MLP uses GELU instead of ReLU²."""

    def __init__(self, config, layer_idx):
        nn.Module.__init__(self)
        from core.model.gpt import CausalSelfAttention
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = GELUMLP(config)


class GELUGPT(GPT):
    """The reference GPT with GELU activation in every MLP.
    Config-compatible with GPT (same GPTConfig)."""

    Config = GPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.transformer.h = nn.ModuleList(
            [GELUBlock(config, i) for i in range(config.n_layer)])

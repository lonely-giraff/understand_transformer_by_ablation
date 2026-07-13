"""Non-uniform MLP depth-profile trunks.

The reference GPT gives every layer the same MLP expansion ratio (4 * d_model).
This module redistributes a fixed MLP parameter budget across depth by giving
each layer its own integer expansion ratio, then exports four named trunk
classes (ascending / descending / hourglass / diamond) that the text
orchestrator can select via ``model.trunk_class``. ``uniform`` stays on the
unmodified reference ``GPT`` (no override).

Only ``transformer.h`` is replaced; embeddings, attention, RoPE, QK norm,
RMSNorm, residuals, the ReLU-squared activation, bias-free linears, the forward
path, and the inherited zero-initialization of every attention/MLP ``c_proj``
all come from core unchanged. No file under ``core/`` or ``modalities/`` is
touched.
"""

from itertools import chain, repeat

import torch.nn as nn
import torch.nn.functional as F

from core.model.gpt import Block, CausalSelfAttention, GPT, GPTConfig, norm

__all__ = [
    "ANCHORS",
    "ratios_for",
    "VariableMLP",
    "VariableBlock",
    "ProfiledGPT",
    "AscendingGPT",
    "DescendingGPT",
    "HourglassGPT",
    "DiamondGPT",
]


# The d6 anchors are the experiment's elementary profiles. Every supported depth
# is a (possibly repeated) expansion of one of these. ``descending`` is the exact
# reverse of ``ascending`` so the paired profiles can never drift apart.
ANCHORS: dict[str, tuple[int, ...]] = {
    "uniform": (4, 4, 4, 4, 4, 4),
    "ascending": (2, 3, 3, 5, 5, 6),
    "hourglass": (6, 4, 2, 2, 4, 6),
    "diamond": (2, 4, 6, 6, 4, 2),
}
ANCHORS["descending"] = tuple(reversed(ANCHORS["ascending"]))


def ratios_for(profile: str, depth: int) -> tuple[int, ...]:
    """Expand a named profile to ``depth`` per-layer MLP expansion ratios.

    The d6 anchors are repeated ``depth // 6`` adjacent times, so the d12 profile
    is the pairwise repetition of the d6 profile and the mean ratio stays exactly
    4. Depths not divisible by 6 are rejected: interpolation would make the
    parameter-equality and profile semantics less transparent.
    """
    if not isinstance(depth, int) or isinstance(depth, bool) or depth <= 0:
        raise ValueError(f"depth must be a positive integer, got {depth!r}")
    if profile not in ANCHORS:
        raise ValueError(f"unknown profile {profile!r}; choose from {tuple(ANCHORS)}")
    if depth % 6:
        raise ValueError(f"non-uniform profile depth must be divisible by 6, got {depth}")
    copies = depth // 6
    ratios = tuple(chain.from_iterable(repeat(ratio, copies) for ratio in ANCHORS[profile]))
    if len(ratios) != depth or any(type(r) is not int or r <= 0 for r in ratios):
        raise ValueError(f"invalid expanded profile {profile}: {ratios}")
    if sum(ratios) != 4 * depth:
        raise ValueError(f"profile {profile} has ratio sum {sum(ratios)}, expected {4 * depth}")
    return ratios


class VariableMLP(nn.Module):
    """Bias-free ReLU-squared MLP with a per-layer expansion ``ratio``.

    Identical to the reference ``MLP`` except the hidden width is ``ratio *
    n_embd`` instead of the fixed ``4 * n_embd``: ``c_fc: d -> r*d`` followed by
    ``relu(x)**2`` and ``c_proj: r*d -> d``, both bias-free. For a bias-free
    MLP the two matrices hold ``d*(r*d) + (r*d)*d = 2*r*d**2`` parameters, so any
    profile whose ratios sum to ``4 * depth`` matches the reference parameter
    count exactly.
    """

    def __init__(self, config: GPTConfig, ratio: int) -> None:
        super().__init__()
        if type(ratio) is not int or ratio <= 0:
            raise ValueError(f"ratio must be a positive integer, got {ratio!r}")
        hidden = ratio * config.n_embd
        self.ratio = ratio
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class VariableBlock(Block):
    """A reference ``Block`` that builds a ``VariableMLP`` instead of ``MLP``.

    ``nn.Module.__init__`` is called directly (not ``super().__init__``) so the
    reference ``Block.__init__`` — which constructs the fixed-width ``MLP`` — is
    skipped. Attention is the untouched ``CausalSelfAttention``; only the MLP
    width differs. The inherited ``Block.forward`` (pre-norm residual path) is
    reused unchanged.
    """

    def __init__(self, config: GPTConfig, layer_idx: int, ratio: int) -> None:
        nn.Module.__init__(self)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = VariableMLP(config, ratio)


class ProfiledGPT(GPT):
    """Base class for non-uniform trunks: build the reference GPT, then swap in
    profile-shaped blocks.

    Calling ``GPT.__init__`` first constructs the uniform reference blocks
    (along with embeddings, token-type embeddings, and rotary buffers); those
    temporary blocks are then replaced by ``VariableBlock``s sized by the
    resolved profile. The final parameter set depends only on the profile. The
    inherited ``init_weights`` zeros every attention and MLP ``c_proj`` (now
    including the variable-width project MLPs), preserving the reference
    identity-start behavior across all residual blocks.
    """

    Config = GPTConfig
    PROFILE: str

    def __init__(self, config: GPTConfig) -> None:
        super().__init__(config)
        ratios = ratios_for(self.PROFILE, config.n_layer)
        self.mlp_ratios = ratios
        self.transformer.h = nn.ModuleList(
            VariableBlock(config, layer_idx, ratio)
            for layer_idx, ratio in enumerate(ratios)
        )


class AscendingGPT(ProfiledGPT):
    PROFILE = "ascending"


class DescendingGPT(ProfiledGPT):
    PROFILE = "descending"


class HourglassGPT(ProfiledGPT):
    PROFILE = "hourglass"


class DiamondGPT(ProfiledGPT):
    PROFILE = "diamond"

"""Learned sequence-to-one projection for final-token prediction."""

import json
import os

import torch
import torch.nn.functional as F

from core.model.heads import LMHead
from core.tokenization.vocab_layout import VocabLayout


class ProjectedSequenceLMHead(LMHead):
    """Pool every causal hidden state, then run the vocabulary head once."""

    def __init__(self, n_embd, vocab_size, softcap=15.0, max_positions=None):
        super().__init__(n_embd, vocab_size, softcap=softcap)
        if max_positions is None:
            max_positions = int(os.environ["NANOINFRA_SEQUENCE_LEN"]) - 1
        self.position_logits = torch.nn.Parameter(torch.empty(max_positions))

    def init_weights(self):
        super().init_weights()
        torch.nn.init.zeros_(self.position_logits)

    def pooling_weights(self, length):
        if length > self.position_logits.numel():
            raise ValueError(
                f"hidden length {length} exceeds projection size "
                f"{self.position_logits.numel()}"
            )
        return self.position_logits[:length].float().softmax(dim=0)

    def loss(self, hidden, targets):
        if not self.training:
            return super().loss(hidden, targets)

        valid = targets != VocabLayout.IGNORE_INDEX
        if not torch.all(valid.sum(dim=1) == 1):
            raise ValueError(
                "ProjectedSequenceLMHead requires exactly one target per sequence"
            )

        selected = targets.gather(1, valid.to(torch.int64).argmax(dim=1, keepdim=True))
        weights = self.pooling_weights(hidden.size(1)).to(dtype=hidden.dtype)
        pooled = torch.einsum("bth,t->bh", hidden, weights)
        logits = self.forward(pooled)
        return F.cross_entropy(logits, selected.squeeze(1))

    @torch.no_grad()
    def projection_summary(self):
        weights = self.pooling_weights(self.position_logits.numel())
        positions = torch.arange(
            weights.numel(), device=weights.device, dtype=weights.dtype
        )
        top_weights, top_positions = weights.topk(min(5, weights.numel()))
        denominator = max(1, weights.numel() - 1)
        return {
            "expected_position_fraction": float(
                (weights * positions).sum() / denominator
            ),
            "last_quarter_mass": float(weights[(3 * weights.numel()) // 4 :].sum()),
            "last_position_weight": float(weights[-1]),
            "top_positions": [int(value) for value in top_positions.cpu()],
            "top_weights": [float(value) for value in top_weights.cpu()],
        }


def print_projection_summary(head):
    print(
        "PROJECTED_POOL_SUMMARY " + json.dumps(head.projection_summary()),
        flush=True,
    )

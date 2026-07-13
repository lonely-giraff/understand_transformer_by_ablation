"""Dependency-light self-test for the masks and projected head."""

import torch

from core.data.supervision import NextTokenPrediction
from core.tokenization.vocab_layout import VocabLayout
from projected_head import ProjectedSequenceLMHead
from supervision import LastPositionPrediction, RandomPositionPrediction


def main():
    tokens = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
    token_types = torch.zeros_like(tokens)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    weights = torch.ones_like(tokens, dtype=torch.float32)

    all_result = NextTokenPrediction().apply(
        tokens, token_types, attention_mask, weights
    )
    last_result = LastPositionPrediction().apply(
        tokens, token_types, attention_mask, weights
    )
    random_a = RandomPositionPrediction(seed=42).apply(
        tokens, token_types, attention_mask, weights
    )
    random_b = RandomPositionPrediction(seed=42).apply(
        tokens, token_types, attention_mask, weights
    )

    assert all_result["targets"].tolist() == [
        [11, 12, 13, 14],
        [21, 22, 23, -1],
    ]
    assert last_result["targets"].tolist() == [
        [-1, -1, -1, 14],
        [-1, -1, 23, -1],
    ]
    assert (
        (last_result["targets"] != VocabLayout.IGNORE_INDEX).sum(dim=1).tolist()
        == [1, 1]
    )
    assert torch.equal(random_a["targets"], random_b["targets"])

    head = ProjectedSequenceLMHead(4, 10, max_positions=4)
    head.init_weights()
    torch.nn.init.normal_(head.lm_head.weight, std=0.02)
    hidden = torch.randn(2, 4, 4, requires_grad=True)
    targets = torch.tensor([[-1, -1, -1, 3], [-1, -1, 7, -1]])
    loss = head.loss(hidden, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert head.position_logits.grad is not None
    assert hidden.grad is not None
    assert torch.allclose(head.pooling_weights(4), torch.full((4,), 0.25))
    print("supervision self-test OK")


if __name__ == "__main__":
    main()

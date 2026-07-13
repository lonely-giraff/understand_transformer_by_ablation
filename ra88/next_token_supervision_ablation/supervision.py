"""Project-local sparse supervision strategies."""

import torch

from core.data.supervision import NextTokenPrediction
from core.tokenization.vocab_layout import VocabLayout


class LastPositionPrediction(NextTokenPrediction):
    """Keep only the final valid next-token target in each sequence."""

    def apply(self, *args, **kwargs):
        result = super().apply(*args, **kwargs)
        targets = result["targets"]
        valid = targets != VocabLayout.IGNORE_INDEX

        positions = torch.arange(targets.size(1), device=targets.device)
        positions = positions.unsqueeze(0).expand_as(targets)
        last = torch.where(valid, positions, -1).max(dim=1).values
        if (last < 0).any():
            raise ValueError("LastPositionPrediction received a row with no valid targets")

        keep = positions == last.unsqueeze(1)
        result["targets"] = torch.where(
            keep,
            targets,
            torch.full_like(targets, VocabLayout.IGNORE_INDEX),
        )
        result["loss_weights"] = keep.to(dtype=torch.float32)
        return result


class RandomPositionPrediction(NextTokenPrediction):
    """Keep one uniformly sampled valid target in each sequence."""

    def __init__(self, seed=42):
        self.seed = seed
        self.generators = {}

    def apply(self, *args, **kwargs):
        result = super().apply(*args, **kwargs)
        targets = result["targets"]
        valid = targets != VocabLayout.IGNORE_INDEX
        if (~valid.any(dim=1)).any():
            raise ValueError("RandomPositionPrediction received a row with no valid targets")

        device_key = str(targets.device)
        if device_key not in self.generators:
            self.generators[device_key] = torch.Generator(
                device=targets.device
            ).manual_seed(self.seed)
        scores = torch.rand(
            targets.shape,
            generator=self.generators[device_key],
            device=targets.device,
        )
        selected = scores.masked_fill(~valid, -1).argmax(dim=1)
        positions = torch.arange(targets.size(1), device=targets.device).unsqueeze(0)
        keep = positions == selected.unsqueeze(1)

        result["targets"] = torch.where(
            keep,
            targets,
            torch.full_like(targets, VocabLayout.IGNORE_INDEX),
        )
        result["loss_weights"] = keep.to(dtype=torch.float32)
        return result

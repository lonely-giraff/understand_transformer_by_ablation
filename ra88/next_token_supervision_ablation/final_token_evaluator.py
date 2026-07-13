"""Shared fixed-context final-token benchmark."""

import math
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from core.evaluation.evaluator import Evaluator
from modalities.text.fineweb import token_data_loader
from modalities.text.tokenizer import get_token_bytes


class FinalTokenEvaluator(Evaluator):
    """Read the same T-token prefix and predict token T+1 for every model."""

    def __init__(self, eval_config, device_batch_size, sequence_len):
        self.interval_steps = eval_config.get("interval_steps", 50)
        eval_at = eval_config.get("eval_at")
        self.eval_at = {int(step) for step in eval_at} if eval_at else None
        self.batch_size = device_batch_size
        self.context_len = sequence_len - 1
        self.eval_sequences = int(
            os.environ.get("NANOINFRA_FINAL_EVAL_SEQUENCES", "2048")
        )
        self.eval_steps = max(1, self.eval_sequences // self.batch_size)
        self.token_bytes = get_token_bytes(device="cuda")

    @torch.no_grad()
    def evaluate(self, model, autocast_ctx):
        loader = token_data_loader(
            B=self.batch_size,
            T=self.context_len,
            split="val",
        )
        total_nats = torch.zeros((), dtype=torch.float32, device="cuda")
        total_bytes = torch.zeros((), dtype=torch.int64, device="cuda")
        total_correct = torch.zeros((), dtype=torch.int64, device="cuda")
        total_count = torch.zeros((), dtype=torch.int64, device="cuda")

        with autocast_ctx:
            for _ in range(self.eval_steps):
                batch = next(loader)
                hidden = model.trunk(
                    batch["idx"], token_types=batch.get("token_types")
                )
                if hasattr(model.head, "pooling_weights"):
                    weights = model.head.pooling_weights(hidden.size(1)).to(hidden.dtype)
                    representation = torch.einsum("bth,t->bh", hidden, weights)
                else:
                    representation = hidden[:, -1]

                logits = model.head(representation)
                targets = batch["targets"][:, -1]
                losses = F.cross_entropy(logits, targets, reduction="none")
                total_nats += losses.sum()
                total_bytes += self.token_bytes[targets].sum()
                total_correct += (logits.argmax(dim=-1) == targets).sum()
                total_count += targets.numel()

        if dist.is_initialized():
            for value in (total_nats, total_bytes, total_correct, total_count):
                dist.all_reduce(value, op=dist.ReduceOp.SUM)

        count = total_count.item()
        byte_count = total_bytes.item()
        return {
            "val/final_ce": total_nats.item() / count,
            "val/final_bpb": total_nats.item() / (math.log(2) * byte_count),
            "val/final_accuracy": total_correct.item() / count,
        }

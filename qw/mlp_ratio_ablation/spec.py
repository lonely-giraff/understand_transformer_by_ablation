"""spec.py — MLP ratio ablation. THE one knob.

Compare MLP expansion ratios: standard 4x vs 2x, 6x, 8x.
Same depth, same dim, same data, same budget — only the trunk class changes,
each pinning a different mlp_ratio.

This tests PARAMETER EFFICIENCY: does expanding the MLP wider buy more
loss reduction than spending the same parameters elsewhere?
"""

DEPTH = 6                  # smoke scale. Raise for a full study.
LR_MAX = "3e-4"
SEED = 42

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS = 20_000_000    # smoke scale — minutes on one GPU
WARMUP_STEPS = 100
N_EVALS = 30               # log-spaced val evals
EVAL_TOKENS = 131072       # 128K val tokens per eval

ORCHESTRATOR = "modalities.text.train_text"

# Four arms — same model geometry, different MLP ratios via trunk_class
# (no core edits: each trunk is a thin subclass of GPT)
ARMS = [
    ("ratio_2x", "projects.mlp_ratio_ablation.trunk.GPT_MLP2x"),
    ("ratio_4x", None),    # None = the reference GPT (4x MLP by default)
    ("ratio_6x", "projects.mlp_ratio_ablation.trunk.GPT_MLP6x"),
    ("ratio_8x", "projects.mlp_ratio_ablation.trunk.GPT_MLP8x"),
]


def train_overrides(trunk_class, max_steps, eval_at):
    """Hydra CLI overrides. The ONLY thing that differs across arms is
    model.trunk_class."""
    ov = {
        "model.depth": DEPTH,
        "optimizer.lr_max": LR_MAX,
        "seed": SEED,
        "sequence_len": SEQ_LEN,
        "device_batch_size": DBS,
        "total_batch_size": TBS,
        "max_steps": max_steps,
        "optimizer.scheduler.warmup_steps": WARMUP_STEPS,
        "optimizer.scheduler.warmdown_ratio": 0.0,   # constant LR
        "optimizer.scheduler.final_lr_frac": 1.0,
        "checkpoint.enabled": "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, eval_at)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 100,
    }
    if trunk_class:
        ov["model.trunk_class"] = trunk_class
    return [f"{k}={v}" for k, v in ov.items()]

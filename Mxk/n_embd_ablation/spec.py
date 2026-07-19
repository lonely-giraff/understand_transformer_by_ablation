"""spec.py — n_embd ablation: how small can the embedding dimension be?"""
DEPTH = 6
LR_MAX = "3e-4"
SEED = 42

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS = 20_000_000
WARMUP_STEPS = 100
N_EVALS = 30
EVAL_TOKENS = 131072

ORCHESTRATOR = "modalities.text.train_text"

ARMS = [
    ("dim=384 (baseline)", None, 384),
    ("dim=64",             None, 64),
    ("dim=16",             None, 16),
]


def train_overrides(trunk_class, max_steps, eval_at, dim):
    ov = {
        "model.depth": DEPTH,
        "model.dim": dim,
        "optimizer.lr_max": LR_MAX,
        "seed": SEED,
        "sequence_len": SEQ_LEN,
        "device_batch_size": DBS,
        "total_batch_size": TBS,
        "max_steps": max_steps,
        "optimizer.scheduler.warmup_steps": WARMUP_STEPS,
        "optimizer.scheduler.warmdown_ratio": 0.0,
        "optimizer.scheduler.final_lr_frac": 1.0,
        "checkpoint.enabled": "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, eval_at)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 100,
    }
    if trunk_class:
        ov["model.trunk_class"] = trunk_class
    return [f"{k}={v}" for k, v in ov.items()]

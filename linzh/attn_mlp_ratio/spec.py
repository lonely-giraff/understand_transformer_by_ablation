"""spec.py — attn:mlp ratio ablation config. THE one knob.

TWO controlled comparisons against the standard GPT baseline:

  * AttnHeavyGPT — more attention blocks, fewer MLP blocks (50-67% attn params)
  * MLPHeavyGPT  — fewer attention blocks, more MLP blocks (11-17% attn params)

ALL three trunks have IDENTICAL total parameter count at every depth
(constraint: 4*N_attn + 8*N_mlp = 12*d). Same data, same budget, same
recipe — only the block layout differs.

Key result (from the full 5-depth scaling study):
  Standard (33% attn) wins at every depth >= 3.
  Frontier exponent a is invariant (~0.5).
"""
DEPTH = 6                    # smoke scale — minutes on one GPU
LR_MAX = "3e-4"              # ONE shared recipe for honest comparison
SEED = 42

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS = 20_000_000
WARMUP_STEPS = 100
N_EVALS = 30
EVAL_TOKENS = 131072

ORCHESTRATOR = "modalities.text.train_text"

# The three arms — (label, trunk import path). None = standard core GPT.
ARMS = [
    ("standard",    None),
    ("attn_heavy",  "variant_trunk.AttnHeavyGPT"),
    ("mlp_heavy",   "variant_trunk.MLPHeavyGPT"),
]


def train_overrides(trunk_class, max_steps, eval_at):
    ov = {
        "model.depth": DEPTH,
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

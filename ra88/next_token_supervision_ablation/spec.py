"""Shared recipe for dense, sparse, and projected next-token supervision."""

DEPTH = 12
LR_MAX = "3e-4"
SEED = 42

SEQ_LEN, DBS, TBS = 512, 32, 16384
MAX_TOKENS = 50_000_000
WARMUP_STEPS = 100
N_EVALS = 20
EVAL_TOKENS = 131072
RUN_TAG = f"d{DEPTH}_{MAX_TOKENS // 1_000_000}m_seed{SEED}"

ORCHESTRATOR = "modalities.text.train_text"

# Standard per-position validation is meaningful for these causal-LM heads.
SUPERVISION_ARMS = [
    ("all_positions", "all"),
    ("random_position", "random"),
    ("last_position", "last"),
]

# These arms share the same sequence-to-one final-token benchmark.
FINAL_TOKEN_ARMS = [
    ("all_positions", "all"),
    ("last_position", "last"),
    ("projected_sequence", "projected"),
]

ALL_ARMS = list(dict(SUPERVISION_ARMS + FINAL_TOKEN_ARMS).items())


def train_overrides(max_steps, eval_at, warmup_steps=WARMUP_STEPS):
    """Hydra overrides held fixed across all arms."""
    overrides = {
        "model.depth": DEPTH,
        "optimizer.lr_max": LR_MAX,
        "seed": SEED,
        "sequence_len": SEQ_LEN,
        "device_batch_size": DBS,
        "total_batch_size": TBS,
        "max_steps": max_steps,
        "optimizer.scheduler.warmup_steps": warmup_steps,
        "optimizer.scheduler.warmdown_ratio": 0.0,
        "optimizer.scheduler.final_lr_frac": 1.0,
        "checkpoint.enabled": "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, eval_at)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 100,
        "wandb.enabled": "false",
    }
    args = [f"{key}={value}" for key, value in overrides.items()]
    # Avoid a fixed trailing EOS target in the literal last-position arm.
    args.append("+data.sources.0.recipe={template:[text_tokens],supervise:all}")
    return args

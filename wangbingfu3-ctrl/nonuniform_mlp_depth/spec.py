"""spec.py — the ablation this project runs. THE one knob.

A controlled depth-allocation ablation: hold everything fixed — total MLP
parameter budget, model geometry, data, optimizer, token budget — and
REDISTRIBUTE that budget across depth. The reference GPT gives every layer the
same MLP expansion ratio (4 * n_embd). Here each layer gets its own integer
ratio, chosen from a named profile, while the ratios are constrained to sum to
``4 * depth`` — so every arm has EXACTLY the same parameter count and FLOPs.
The only thing that differs between arms is WHERE along depth the MLP capacity
sits.

  * uniform     — the reference GPT (every layer 4x)        [baseline]
  * ascending   — capacity skewed toward the TOP (last layers)
  * descending  — capacity skewed toward the BOTTOM (first layers)
  * hourglass   — heavy at the two ends, light in the middle
  * diamond     — heavy in the MIDDLE, light at the two ends

All five are driven through the blessed text Orchestrator
(``modalities.text.train_text``) via its ``model.trunk_class`` knob — no core
edit, no forked training loop. ``uniform`` omits the override and runs the
unmodified reference ``GPT``.

NOTE on budget (read README.md before quoting a number). The curves shipped in
``results/curves.json`` were trained at 1.31B tokens per arm (~8 tok/param) —
about 0.4x the Chinchilla-optimal budget for this 162M model, and a SINGLE seed.
``MAX_TOKENS`` below is the budget the shipped curves used, not the
compute-optimal one.
"""
DEPTH = 12                 # n_layer; n_embd = DEPTH*64 = 768. Must be divisible by 6
                           #   (trunk contract: profiles are d6 anchors repeated).
LR_MAX = "3e-4"
SEED = 42

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS = 1_310_720_000   # what the shipped curves used (~0.4x Chinchilla; see README).
                             #   Chinchilla-optimal would be 20 * 162,203,904 = 3,244,078,080.
WARMUP_STEPS = 100
N_EVALS = 30                 # log-spaced val evals; the first lands early (~step 5)
EVAL_TOKENS = 131072         # 128K val tokens/eval — only relative CE matters here

ORCHESTRATOR = "modalities.text.train_text"

# The five arms: (label, trunk import path).  None => the reference GPT.
ARMS = [
    ("uniform",    None),
    ("ascending",  "trunk.AscendingGPT"),
    ("descending", "trunk.DescendingGPT"),
    ("hourglass",  "trunk.HourglassGPT"),
    ("diamond",    "trunk.DiamondGPT"),
]


def train_overrides(trunk_class, max_steps, eval_at):
    """Hydra CLI overrides — this project's recipe on the orchestrator's defaults.
    Constant LR (no warmdown) so each curve is genuine loss-vs-step; an explicit
    log-spaced eval schedule; no checkpoints (the shipped curves are what matter).
    The ONLY thing that changes between arms is ``model.trunk_class``."""
    ov = {
        "model.depth": DEPTH,
        "optimizer.lr_max": LR_MAX,
        "seed": SEED,
        "sequence_len": SEQ_LEN,
        "device_batch_size": DBS,
        "total_batch_size": TBS,
        "max_steps": max_steps,
        "optimizer.scheduler.warmup_steps": WARMUP_STEPS,
        "optimizer.scheduler.warmdown_ratio": 0.0,   # constant LR after warmup
        "optimizer.scheduler.final_lr_frac": 1.0,
        "checkpoint.enabled": "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, eval_at)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 50,
        "wandb.enabled": "false",
    }
    if trunk_class:
        ov["model.trunk_class"] = trunk_class
    return [f"{k}={v}" for k, v in ov.items()]

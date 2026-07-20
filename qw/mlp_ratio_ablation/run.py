"""run.py — train all mlp_ratio arms and collect val-CE trajectories.

Drives the blessed orchestrator via subprocess. Same model geometry / data /
budget; the ONLY difference is model.trunk_class — each trunk pins a different
MLP expansion ratio.

Usage:
    python projects/mlp_ratio_ablation/run.py      # trains all arms
    python projects/mlp_ratio_ablation/plot.py     # -> mlp_ratio_ablation.png
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import spec

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")


def eval_schedule(max_steps, n=spec.N_EVALS, first=5):
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def count_params(depth, dim, mlp_ratio, vocab_size=50304):
    """Estimate non-embedding params for given geometry + mlp_ratio."""
    attn_per_layer = 4 * dim * dim          # c_q, c_k, c_v, c_proj
    mlp_per_layer = 2 * dim * int(dim * mlp_ratio)  # c_fc + c_proj
    n_layer_params = depth * (attn_per_layer + mlp_per_layer)
    return n_layer_params + 3 * dim          # final norm


def run_arm(label, trunk_class, mlp_ratio, max_steps, steps):
    ov = spec.train_overrides(trunk_class, max_steps, steps)
    print(f"[run ] {label} (ratio={mlp_ratio}x, trunk={trunk_class or 'GPT'}): "
          f"d{spec.DEPTH} -> {max_steps} steps ...", flush=True)
    out = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                         cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)},
                         capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    traj = [{"step": int(s), "val": float(v)} for s, v in EVAL_RE.findall(text)]
    if out.returncode != 0 or len(traj) < 3:
        raise SystemExit(
            f"arm {label} FAILED (rc={out.returncode}, {len(traj)} evals):\n{text[-3000:]}")
    print(f"[done] {label}: {len(traj)} evals, "
          f"val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}", flush=True)
    return {"arm": label, "mlp_ratio": mlp_ratio, "trunk_class": trunk_class,
            "trajectory": traj}


def main():
    dim = spec.DEPTH * 64
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)

    # label -> mlp_ratio mapping (trunk class → the ratio it pins)
    RATIO_MAP = {"ratio_2x": 2.0, "ratio_4x": 4.0, "ratio_6x": 6.0, "ratio_8x": 8.0}

    arms_data = []
    for label, trunk_class in spec.ARMS:
        ratio = RATIO_MAP[label]
        arm = run_arm(label, trunk_class, ratio, max_steps, steps)
        N = count_params(spec.DEPTH, dim, ratio)
        arm["N"] = N
        arm["dim"] = dim
        arms_data.append(arm)

    out = {"depth": spec.DEPTH, "dim": dim,
           "max_steps": max_steps, "arms": arms_data}
    (RESULTS / "curves.json").write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {RESULTS / 'curves.json'}")


if __name__ == "__main__":
    main()

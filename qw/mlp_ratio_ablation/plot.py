"""plot.py — MLP ratio ablation: loss curves + parameter-efficiency comparison.

Produces mlp_ratio_ablation.png with two panels:
  left:  val CE vs step (which ratio learns best on a fixed token budget?)
  right: val CE vs non-embedding params (parameter efficiency)

Usage:
    python projects/mlp_ratio_ablation/plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

COLORS = {
    "ratio_2x": "#e41a1c",
    "ratio_4x": "#377eb8",
    "ratio_6x": "#4daf4a",
    "ratio_8x": "#984ea3",
}
TBS = 16384  # must match spec.py


def load():
    curves = json.loads((RESULTS / "curves.json").read_text())
    return curves


def plot(curves_data):
    arms = curves_data["arms"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ---- left: CE vs tokens (steps * TBS) ----
    for arm in arms:
        label = arm["arm"]
        traj = arm["trajectory"]
        xs = [p["step"] * TBS / 1e6 for p in traj]
        ys = [p["val"] for p in traj]
        ax1.plot(xs, ys, "-o", color=COLORS.get(label, "#888"),
                 lw=1.5, ms=4, label=f'{label} (ratio={arm["mlp_ratio"]})')

    ax1.set_xlabel("Training Tokens (millions)")
    ax1.set_ylabel("Validation CE")
    ax1.set_title(f"MLP Ratio Ablation — d{curves_data['depth']}, "
                  f"dim={curves_data['dim']}\n"
                  f"fixed budget {curves_data['max_steps']*TBS/1e6:.0f}M tokens")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ---- right: final CE vs non-embedding params ----
    final_ce = [arm["trajectory"][-1]["val"] for arm in arms]
    params_M = [arm["N"] / 1e6 for arm in arms]
    ratios = [arm["mlp_ratio"] for arm in arms]

    for i, arm in enumerate(arms):
        ax2.plot(params_M[i], final_ce[i], "o", color=COLORS.get(arm["arm"], "#888"),
                 ms=12, label=f'{arm["arm"]} ({arm["mlp_ratio"]}x)')
        ax2.annotate(f"{arm['mlp_ratio']}x", (params_M[i], final_ce[i]),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=9, ha="center", color=COLORS.get(arm["arm"], "#888"))

    ax2.set_xlabel("Non-embedding Parameters (millions)")
    ax2.set_ylabel("Final Validation CE")
    ax2.set_title(f"Parameter Efficiency — d{curves_data['depth']}, "
                  f"dim={curves_data['dim']}")
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # lower CE = better

    # ---- text summary box ----
    best_ce = min(final_ce)
    best_idx = final_ce.index(best_ce)
    best_label = arms[best_idx]["arm"]

    summary = (
        f"Best final CE: {best_label} ({arms[best_idx]['mlp_ratio']}x) = {best_ce:.4f}\n"
        f"Baseline (4x) params: {params_M[1]:.2f}M  CE: {final_ce[1]:.4f}\n"
    )
    for i, arm in enumerate(arms):
        if arm["arm"] != "ratio_4x":
            d_ce = final_ce[i] - final_ce[1]
            d_p = (params_M[i] - params_M[1]) / params_M[1] * 100
            summary += (
                f"{arm['arm']}: ΔCE={d_ce:+.4f}  Δparams={d_p:+.0f}%\n"
            )

    ax2.text(0.95, 0.05, summary.strip(), transform=ax2.transAxes,
             fontsize=8, fontfamily="monospace", va="bottom", ha="right",
             bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))

    fig.tight_layout()
    outpath = HERE / "mlp_ratio_ablation.png"
    fig.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")
    print(summary)


if __name__ == "__main__":
    data = load()
    plot(data)

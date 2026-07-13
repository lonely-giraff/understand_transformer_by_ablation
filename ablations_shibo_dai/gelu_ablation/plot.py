"""plot.py — two val-loss curves, ReLU² vs GELU → gelu_ablation.png"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "results" / "curves.json").read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "baseline": ("#1f77b4", "baseline (ReLU²)"),
        "gelu":     ("#d62728", "GELU"),
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for arm in data["arms"]:
        tr = arm["trajectory"]
        color, label = style.get(arm["arm"], ("gray", arm["arm"]))
        ax.plot([p["step"] for p in tr], [p["val"] for p in tr], "-o",
                color=color, lw=1.9, ms=4, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation cross-entropy")
    ax.set_title(f"ReLU² vs GELU activation  (d{data['depth']})\n"
                 "same model, data, budget — only the activation function changed")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = HERE / "gelu_ablation.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

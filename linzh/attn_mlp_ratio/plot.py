"""plot.py — attn:mlp ratio ablation: three val-loss curves from run.py's
results/curves.json. Writes attn_mlp_ratio.png."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "results" / "curves.json").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {
        "standard":   ("#2c7bb6", "Standard  (33% attn params)"),
        "attn_heavy": ("#fdae61", "Attn-Heavy  (~56% attn params)"),
        "mlp_heavy":  ("#d7191c", "MLP-Heavy  (~11% attn params)"),
    }
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    for arm in data["arms"]:
        tr = arm["trajectory"]
        color, label = style.get(arm["arm"], ("gray", arm["arm"]))
        ax.plot([p["step"] for p in tr], [p["val"] for p in tr], "-o",
                color=color, lw=1.9, ms=3, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("validation cross-entropy")
    ax.set_title(
        f"Attn:MLP ratio ablation  (d{data['depth']}, same total params)\n"
        "Standard (33% attn) wins — don't mess with the ratio"
    )
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    out = HERE / "attn_mlp_ratio.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

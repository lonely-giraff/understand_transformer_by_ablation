"""plot.py — the five val-CE curves (one per MLP depth profile) and the
per-layer expansion-ratio profile of each arm, from results/curves.json.

Writes:
  nonuniform_mlp.png       — validation cross-entropy vs training tokens (hero)
  figures/profiles.png     — per-layer MLP expansion ratio per arm

Self-contained: reads only results/curves.json. Profile ratios are embedded in
that file, so this script does not import trunk.py and needs no nanoinfra/torch.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

STYLE = {
    "uniform":    ("#1f77b4", "uniform (reference)"),
    "ascending":  ("#2ca02c", "ascending (top-heavy)"),
    "descending": ("#d62728", "descending (bottom-heavy)"),
    "hourglass":  ("#9467bd", "hourglass (ends-heavy)"),
    "diamond":    ("#ff7f0e", "diamond (middle-heavy)"),
}


def main():
    data = json.loads((HERE / "results" / "curves.json").read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depth = data["depth"]

    # --- hero figure: val CE vs tokens, all five arms ---
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    for arm in data["arms"]:
        tr = arm["trajectory"]
        color, label = STYLE.get(arm["arm"], ("gray", arm["arm"]))
        ax.plot([p["tokens"] for p in tr], [p["val"] for p in tr], "-o",
                color=color, lw=1.9, ms=3.5,
                label=f"{label}  — final {arm['final_val']:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("training tokens")
    ax.set_ylabel("validation cross-entropy")
    ax.set_title(
        f"Non-uniform MLP depth allocation  (d{depth}, {data['max_tokens']:,} tokens/arm, seed {data['seed']})\n"
        "identical parameter count and FLOPs — only the per-layer MLP ratio differs")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = HERE / "nonuniform_mlp.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # --- profile figure: per-layer expansion ratio per arm ---
    (HERE / "figures").mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    width = 0.16
    layers = list(range(depth))
    order = ["uniform", "ascending", "descending", "hourglass", "diamond"]
    for i, name in enumerate(order):
        arm = next(a for a in data["arms"] if a["arm"] == name)
        ratios = arm["profile"]
        color, _ = STYLE[name]
        ax.bar([l + (i - 2) * width for l in layers], ratios, width=width,
               color=color, label=name)
    ax.set_xlabel("layer index (0 = first / bottom)")
    ax.set_ylabel("MLP expansion ratio (x n_embd)")
    ax.set_title(f"Per-layer MLP expansion ratio by profile  (d{depth}; "
                 f"each arm sums to {4 * depth} = 4·{depth}, so params are equal)")
    ax.set_xticks(layers)
    ax.legend(fontsize=9, ncol=5)
    fig.tight_layout()
    out = HERE / "figures" / "profiles.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

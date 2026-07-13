"""Render the standard-supervision and shared final-token comparisons."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

STYLES = {
    "all_positions": ("#1769aa", "all positions"),
    "random_position": ("#2a9d58", "one random position"),
    "last_position": ("#d1495b", "last position only"),
    "projected_sequence": ("#7b4ab5", "learned sequence projection"),
}


def _plot(data, ylabel, title, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for arm in data["arms"]:
        color, label = STYLES[arm["arm"]]
        trajectory = arm["trajectory"]
        ax.plot(
            [point["step"] for point in trajectory],
            [point["val"] for point in trajectory],
            "-o",
            color=color,
            linewidth=1.9,
            markersize=4,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"wrote {output}")


def main():
    FIGURES.mkdir(exist_ok=True)
    supervision = json.loads(
        (RESULTS / "supervision_d12_50m.json").read_text()
    )
    final_token = json.loads(
        (RESULTS / "shared_final_token_d12_50m.json").read_text()
    )
    _plot(
        supervision,
        "all-position validation cross-entropy",
        "Supervision density ablation (d12)\n"
        "same trunk, data, and 50M-token budget",
        FIGURES / "supervision_ablation.png",
    )
    _plot(
        final_token,
        "final-token validation cross-entropy",
        "Can one sequence-level loss replace dense supervision? (d12)\n"
        "same 511-token prefix and held-out next token",
        FIGURES / "shared_final_token_comparison.png",
    )


if __name__ == "__main__":
    main()

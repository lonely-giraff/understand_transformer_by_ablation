"""plot.py — three learning curves: dim=384 vs dim=64 vs dim=16."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = json.load(open(HERE / "results" / "curves.json"))

colors = ["#1f77b4", "#ff7f0e", "#d62728"]

fig, ax = plt.subplots(figsize=(8, 5))
for arm, c in zip(data["arms"], colors):
    xs = [p["step"] for p in arm["trajectory"]]
    ys = [p["val"] for p in arm["trajectory"]]
    ax.plot(xs, ys, color=c, lw=1.5, label=arm["arm"])

ax.set_xlabel("Step")
ax.set_ylabel("Validation CE (nats)")
ax.set_title(f"n_embd ablation — depth={data['depth']}, modern trunk")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "n_embd_ablation.png", dpi=150)
print(f"-> {HERE / 'n_embd_ablation.png'}")

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

EXP_DIR = Path(__file__).parent / "experiments"

def load(run_id):
    p = EXP_DIR / run_id / "metrics.jsonl"
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip().replace("'", '"')
        if line:
            rows.append(json.loads(line))
    return rows

# ── data ──────────────────────────────────────────────────────────────────────
wnorm_runs = {
    "SwiGLU + qknorm + wnorm (best)":       "20260328_180644_370333_standard_swiglu_qknorm_wnorm_d256",
    "PFDRationalGLU + qknorm + wnorm":      "20260327_214545_306009_standard_pfd_rationalglu_qknorm_wnorm_d256",
    "SwiGLU + qknorm + wnorm (early run)":  "20260324_164546_standard_swiglu_qknorm_wnorm_d256",
    "XSA + qknorm + wnorm + ortho":         "20260416_105724_190084_xsa_swiglu_qknorm_wnorm_orthogonal_d256",
}

no_wnorm_runs = {
    "SwiGLU baseline":                      "20260316_135157_standard_swiglu_d256",
    "SwiGLU + qknorm":                      "20260318_105528_standard_swiglu_qknorm_d256",
    "XSA + SwiGLU":                         "20260415_123631_662081_xsa_swiglu_d256",
    "XSA + SwiGLU + qknorm":                "20260415_145723_799222_xsa_swiglu_qknorm_d256",
    "XSA + qknorm + ortho FFN (best no-wn)":"20260415_231629_387840_xsa_swiglu_qknorm_orthogonal_d256",
}

wnorm_colors = [
    "#1565C0",  # deep blue
    "#6A1B9A",  # deep purple
    "#0097A7",  # teal
    "#2E7D32",  # dark green
]
no_wnorm_colors = [
    "#EF9A9A",  # light red
    "#FFCC80",  # light orange
    "#80DEEA",  # light cyan
    "#A5D6A7",  # light green
    "#CE93D8",  # light purple
]

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), sharey=False)
fig.suptitle(
    "Weight Norm vs No-Weight-Norm: Training & Validation Curves\n"
    "(WikiText-103, d_model=256, BPE-50257)",
    fontsize=13, fontweight="bold", y=1.02
)

panels = [
    ("train_loss", "Train Loss (cross-entropy)", None),
    ("val_ppl",    "Validation Perplexity",       None),
]

for ax_idx, (metric, ylabel, _) in enumerate(panels):
    ax = axes[ax_idx]
    ax.set_title(ylabel, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    for i, (name, run_id) in enumerate(wnorm_runs.items()):
        data = load(run_id)
        epochs = [d["epoch"] for d in data]
        values = [d[metric] for d in data]
        color = wnorm_colors[i % len(wnorm_colors)]
        line, = ax.plot(epochs, values, color=color, linewidth=2.2,
                        linestyle="-", marker="o", markersize=5, zorder=3)
        # annotate final value
        ax.annotate(f"{values[-1]:.2f}", xy=(epochs[-1], values[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7.5, color=color, va="center")

    for i, (name, run_id) in enumerate(no_wnorm_runs.items()):
        data = load(run_id)
        epochs = [d["epoch"] for d in data]
        values = [d[metric] for d in data]
        color = no_wnorm_colors[i % len(no_wnorm_colors)]
        line, = ax.plot(epochs, values, color=color, linewidth=1.8,
                        linestyle="--", marker="s", markersize=4, zorder=2)
        ax.annotate(f"{values[-1]:.2f}", xy=(epochs[-1], values[-1]),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=7.5, color=color, va="center")

    ax.set_xticks(range(10))
    ax.grid(True, alpha=0.3, linestyle=":")

    # clip epoch-0 train spike for better readability
    if metric == "train_loss":
        ax.set_ylim(3.7, 8.5)
    elif metric == "val_ppl":
        ax.set_ylim(35, 150)

    # shade the wnorm "advantage zone"
    if metric == "val_ppl":
        ax.axhline(75.68, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
        ax.text(9.1, 75.68, "SwiGLU\nbaseline", fontsize=7, color="gray", va="center")

# ── shared legend ──────────────────────────────────────────────────────────────
wnorm_patch = mpatches.Patch(color="#1565C0", label="── Weight Norm (solid)")
no_wnorm_patch = mpatches.Patch(color="#EF9A9A", label="-- No Weight Norm (dashed)")

handles_w = [
    plt.Line2D([0], [0], color=c, lw=2, ls="-", marker="o", ms=5, label=n)
    for (n, _), c in zip(wnorm_runs.items(), wnorm_colors)
]
handles_nw = [
    plt.Line2D([0], [0], color=c, lw=1.8, ls="--", marker="s", ms=4, label=n)
    for (n, _), c in zip(no_wnorm_runs.items(), no_wnorm_colors)
]

fig.legend(
    handles=handles_w + handles_nw,
    loc="lower center",
    ncol=3,
    fontsize=8.5,
    framealpha=0.9,
    bbox_to_anchor=(0.5, -0.18),
)

plt.tight_layout()
out = Path(__file__).parent / "wnorm_loss_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

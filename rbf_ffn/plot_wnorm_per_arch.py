import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# Each entry: (title, no_wnorm_runs, wnorm_runs)
# no_wnorm_runs / wnorm_runs: list of (label, run_id)
ARCHITECTURES = [
    (
        "SwiGLU\n(Standard Attn)",
        [("no wnorm", "20260316_135157_standard_swiglu_d256")],
        [("+ wnorm",  "20260324_153809_standard_swiglu_wnorm_d256")],
    ),
    (
        "SwiGLU + QK Norm\n(Standard Attn)",
        [("no wnorm", "20260318_105528_standard_swiglu_qknorm_d256")],
        [("+ wnorm",  "20260328_180644_370333_standard_swiglu_qknorm_wnorm_d256")],
    ),
    (
        "XSA + SwiGLU + QK Norm",
        [("no wnorm (run A)", "20260415_145723_799222_xsa_swiglu_qknorm_d256"),
         ("no wnorm (run B)", "20260506_130221_500343_xsa_swiglu_qknorm_d256")],
        [("+ wnorm (0430)",   "20260430_132442_213125_xsa_swiglu_qknorm_wnorm_d256"),
         ("+ wnorm (0509)",   "20260509_090930_507143_xsa_swiglu_qknorm_wnorm_d256"),
         ("+ wnorm (0513)",   "20260513_101900_905014_xsa_swiglu_qknorm_wnorm_d256")],
    ),
    (
        "XSA + QK Norm\n+ Orthogonal FFN",
        [("no wnorm", "20260415_231629_387840_xsa_swiglu_qknorm_orthogonal_d256")],
        [("+ wnorm",  "20260416_105724_190084_xsa_swiglu_qknorm_wnorm_orthogonal_d256")],
    ),
    (
        "XSA + MoE\n+ QK Norm",
        [("no wnorm",        "20260504_225655_397364_xsa_moe_qknorm_d256")],
        [("+ wnorm",         "20260508_085651_652695_xsa_moe_qknorm_wnorm_d256"),
         ("+ wnorm dyn-erf", "20260508_120315_709553_xsa_moe_qknorm_wnorm_dynamic_erf_d256"),
         ("+ wnorm orth",    "20260509_110833_209194_xsa_moe_qknorm_wnorm_orthogonal_d256")],
    ),
    (
        "XSA + ReLU² + QK Norm",
        [("leaky-relu² (run A)", "20260415_180252_016541_xsa_leaky_relu_sq_qknorm_d256"),
         ("leaky-relu² (run B)", "20260415_190149_519409_xsa_leaky_relu_sq_qknorm_d256")],
        [("+ relu² wnorm",       "20260515_155952_271445_xsa_relu_sq_qknorm_wnorm_d256")],
    ),
]

BLUE_SHADES   = ["#1565C0", "#1E88E5", "#42A5F5", "#90CAF9"]   # wnorm  — solid
ORANGE_SHADES = ["#E65100", "#FB8C00", "#FFCC80"]              # no wnorm — dashed

N = len(ARCHITECTURES)

fig = plt.figure(figsize=(24, 9))
fig.suptitle(
    "Effect of Weight Norm by Architecture — Train Loss & Validation PPL\n"
    "(WikiText-103, d_model=256)",
    fontsize=13, fontweight="bold", y=1.02,
)

# 2 rows (train / val PPL) × N cols
gs = gridspec.GridSpec(2, N, figure=fig, hspace=0.5, wspace=0.35)

for col, (title, no_wn, wn) in enumerate(ARCHITECTURES):
    for row, (metric, ylabel) in enumerate([("train_loss", "Train Loss"), ("val_ppl", "Val PPL")]):
        ax = fig.add_subplot(gs[row, col])

        if row == 0:
            ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8.5)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, linestyle=":")

        # no-wnorm runs (dashed)
        for k, (label, run_id) in enumerate(no_wn):
            data = load(run_id)
            epochs = [d["epoch"] for d in data]
            vals   = [d[metric]  for d in data]
            color  = ORANGE_SHADES[k % len(ORANGE_SHADES)]
            ax.plot(epochs, vals, color=color, lw=1.8, ls="--",
                    marker="s", ms=4, label=label, zorder=2)
            ax.annotate(f"{vals[-1]:.2f}", xy=(epochs[-1], vals[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=7.5, color=color, va="center", fontweight="bold")

        # wnorm runs (solid)
        for k, (label, run_id) in enumerate(wn):
            data = load(run_id)
            epochs = [d["epoch"] for d in data]
            vals   = [d[metric]  for d in data]
            color  = BLUE_SHADES[k % len(BLUE_SHADES)]
            ax.plot(epochs, vals, color=color, lw=2.2, ls="-",
                    marker="o", ms=4, label=label, zorder=3)
            ax.annotate(f"{vals[-1]:.2f}", xy=(epochs[-1], vals[-1]),
                        xytext=(4, 0), textcoords="offset points",
                        fontsize=7.5, color=color, va="center", fontweight="bold")

        ax.set_xticks(range(0, 10, 2))

        # sensible y-limits
        if metric == "train_loss":
            ax.set_ylim(3.6, 8.8)
        else:
            ax.set_ylim(30, 155)

        if col == N - 1:  # last column — add legend inside
            if row == 0:
                ax.legend(fontsize=6.5, loc="upper right", framealpha=0.8)

# global legend patches
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], color=BLUE_SHADES[0],   lw=2.2, ls="-",  marker="o", ms=5, label="with weight norm (solid)"),
    Line2D([0], [0], color=ORANGE_SHADES[0], lw=1.8, ls="--", marker="s", ms=4, label="no weight norm (dashed)"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
out = Path(__file__).parent / "wnorm_per_arch.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")

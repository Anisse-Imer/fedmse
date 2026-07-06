"""
Visualization script for centralized training results.
Auto-discovers all runs under Checkpoint/Results/Centralized/ and produces
comparison charts across distributions (IID, nonIID) and model types.

Figures produced:
  1. AUC over phases — one subplot per model type, IID vs nonIID overlaid
  2. Per-client best-phase AUC — grouped bars across all combinations
  3. Validation loss curves — one subplot per model type, distributions overlaid
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = "Checkpoint/Results/Centralized"
OUT_DIR  = "Visualization/Images"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG      = "#0B0D14"
SURFACE = "#141720"
BORDER  = "#1E2130"
GRID    = "#1A1D2B"
TEXT    = "#EDEFF5"
MUTED   = "#5A5F7A"

# Per (distribution, model_type) colour + style
STYLES = {
    ("IID",    "autoencoder"): {"color": "#4E9CF6", "ls": "-",  "marker": "o", "label": "IID · Autoencoder"},
    ("IID",    "hybrid"):      {"color": "#2ECC8E", "ls": "-",  "marker": "s", "label": "IID · Hybrid"},
    ("nonIID", "autoencoder"): {"color": "#F97316", "ls": "--", "marker": "o", "label": "nonIID · Autoencoder"},
    ("nonIID", "hybrid"):      {"color": "#C084FC", "ls": "--", "marker": "s", "label": "nonIID · Hybrid"},
}

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        GRID,
    "grid.linewidth":    0.6,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "font.size":         10,
    "legend.facecolor":  SURFACE,
    "legend.edgecolor":  BORDER,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def infer_distribution(exp_dir_name):
    if "nonIID" in exp_dir_name or "non-iid" in exp_dir_name.lower():
        return "nonIID"
    return "IID"

def load_result_file(path):
    """Return (phases, clients_per_phase, val_losses, train_losses)."""
    phases, clients_per_phase, val_losses, train_losses = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d    = json.loads(line)
            key  = next(iter(d))
            data = d[key]
            phases.append(int(key.split("_")[1]))
            clients_per_phase.append({k: v for k, v in data.items()
                                       if k not in ("val_loss", "train_loss")})
            val_losses.append(data["val_loss"])
            train_losses.append(data["train_loss"])
    return phases, clients_per_phase, val_losses, train_losses

def mean_std(clients_per_phase):
    means, stds = [], []
    for cp in clients_per_phase:
        vals = list(cp.values())
        means.append(np.mean(vals))
        stds.append(np.std(vals))
    return np.array(means), np.array(stds)

# ---------------------------------------------------------------------------
# Discover all runs
# ---------------------------------------------------------------------------
runs = []   # list of dicts with all data for one (distribution, model_type) combo

for exp_dir in sorted(glob.glob(os.path.join(BASE_DIR, "*", "*"))):
    if not os.path.isdir(exp_dir):
        continue
    exp_name     = os.path.basename(exp_dir)   # e.g. Centralized_nonIID_100epoch_...
    distribution = infer_distribution(exp_name)

    for model_type in ("autoencoder", "hybrid"):
        result_path = os.path.join(exp_dir, "AUC", f"Centralized_{model_type}_results.json")
        if not os.path.exists(result_path):
            continue
        phases, clients_per_phase, val_losses, train_losses = load_result_file(result_path)
        if not phases:
            continue
        m, s = mean_std(clients_per_phase)
        runs.append({
            "distribution":      distribution,
            "model_type":        model_type,
            "phases":            phases,
            "clients_per_phase": clients_per_phase,
            "val_losses":        val_losses,
            "train_losses":      train_losses,
            "mean_auc":          m,
            "std_auc":           s,
        })
        print(f"Loaded: {distribution:6s} · {model_type:12s} — {len(phases)} phases")

if not runs:
    raise FileNotFoundError(f"No result files found under {BASE_DIR}")

# Collect all client names (consistent order)
client_names = sorted(
    runs[0]["clients_per_phase"][0].keys(),
    key=lambda x: int(x.split("-")[-1])
)
short_names = [f"C{int(c.split('-')[-1])}" for c in client_names]

# ---------------------------------------------------------------------------
# Figure 1 — AUC over phases (one subplot per model type)
# ---------------------------------------------------------------------------
model_types = sorted({r["model_type"] for r in runs})
fig, axes = plt.subplots(1, len(model_types), figsize=(5 * len(model_types), 4.5), sharey=False)
if len(model_types) == 1:
    axes = [axes]
fig.suptitle("AUC over Training Phases — Centralized", fontsize=11, y=1.01)

for ax, mt in zip(axes, model_types):
    mt_runs = [r for r in runs if r["model_type"] == mt]
    for r in mt_runs:
        style = STYLES.get((r["distribution"], r["model_type"]),
                           {"color": "#888", "ls": "-", "marker": "o", "label": r["distribution"]})
        phases = r["phases"]
        m, s   = r["mean_auc"], r["std_auc"]

        # Per-client faint traces
        for c in client_names:
            ax.plot(phases, [cp[c] for cp in r["clients_per_phase"]],
                    color=style["color"], alpha=0.1, linewidth=0.7)

        # Std band + mean line
        ax.fill_between(phases, m - s, m + s, color=style["color"], alpha=0.15)
        ax.plot(phases, m, color=style["color"], linewidth=2.2,
                linestyle=style["ls"], marker=style["marker"],
                markersize=4.5, label=style["label"])

    ax.set_title(mt.capitalize(), fontsize=10)
    ax.set_xlabel("Phase", labelpad=6)
    ax.set_ylabel("AUC", labelpad=6)
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.grid(axis="y")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/centralized_auc_phases.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/centralized_auc_phases.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: centralized_auc_phases")

# ---------------------------------------------------------------------------
# Figure 2 — Per-client best-phase AUC grouped bar chart
# ---------------------------------------------------------------------------
# For each run pick the phase with the highest mean AUC
run_bar_data = []
for r in runs:
    best_idx  = int(np.argmax(r["mean_auc"]))
    best_phase = r["phases"][best_idx]
    auc_vals   = [r["clients_per_phase"][best_idx][c] for c in client_names]
    style      = STYLES.get((r["distribution"], r["model_type"]),
                             {"color": "#888", "label": r["distribution"]})
    run_bar_data.append({
        "label":      style["label"],
        "color":      style["color"],
        "auc_vals":   auc_vals,
        "best_phase": best_phase,
    })

n_clients = len(client_names)
n_runs    = len(run_bar_data)
group_w   = 0.8
bar_w     = group_w / n_runs
x         = np.arange(n_clients)

fig, ax = plt.subplots(figsize=(13, 4.5))
for i, rd in enumerate(run_bar_data):
    offsets = x - group_w / 2 + bar_w * i + bar_w / 2
    ax.bar(offsets, rd["auc_vals"], width=bar_w * 0.88,
           color=rd["color"], alpha=0.85,
           label=f"{rd['label']} (ph. {rd['best_phase']})")

ax.set_xticks(x)
ax.set_xticklabels(short_names, fontsize=9)
ax.set_ylabel("AUC", labelpad=8)
ax.set_title("Per-Client AUC at Best Phase — IID vs nonIID", pad=12, fontsize=11)
ax.set_ylim(0.88, 1.01)
ax.yaxis.set_major_locator(MultipleLocator(0.02))
ax.grid(axis="y")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/centralized_auc_per_client.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/centralized_auc_per_client.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: centralized_auc_per_client")

# ---------------------------------------------------------------------------
# Figure 3 — Validation loss curves (one subplot per model type)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(model_types), figsize=(5 * len(model_types), 4), sharey=False)
if len(model_types) == 1:
    axes = [axes]
fig.suptitle("Validation Loss — Centralized", fontsize=11, y=1.01)

for ax, mt in zip(axes, model_types):
    for r in [r for r in runs if r["model_type"] == mt]:
        style = STYLES.get((r["distribution"], r["model_type"]),
                           {"color": "#888", "ls": "-", "marker": "o", "label": r["distribution"]})
        ax.plot(r["phases"], r["val_losses"],
                color=style["color"], linewidth=2,
                linestyle=style["ls"], marker=style["marker"],
                markersize=4, label=style["label"])
    ax.set_title(mt.capitalize(), fontsize=10)
    ax.set_xlabel("Phase", labelpad=6)
    ax.set_ylabel("Loss", labelpad=6)
    ax.grid(axis="y")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/centralized_loss_curves.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/centralized_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: centralized_loss_curves")

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n── Best mean AUC per run ──────────────────────────────")
print(f"{'Distribution':10s}  {'Model':12s}  {'Best AUC':>9s}  {'Phase':>6s}  {'Phases run':>10s}")
print("─" * 56)
for r in sorted(runs, key=lambda r: (r["distribution"], r["model_type"])):
    best_idx = int(np.argmax(r["mean_auc"]))
    print(f"{r['distribution']:10s}  {r['model_type']:12s}  "
          f"{r['mean_auc'][best_idx]:.4f}     ph.{r['phases'][best_idx]:2d}  "
          f"{len(r['phases']):>10d}")

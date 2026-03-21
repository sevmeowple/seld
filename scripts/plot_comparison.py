"""Generate comparison charts for models with complete train+test results."""
import tomli
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

with open("experiments/metrics.toml", "rb") as f:
    data = tomli.load(f)

MODELS = {
    "u2_dynamic_chunk": "U2-DC",
    "hoom": "HOOM",
    "hoom_fix": "HOOM-fix",
    "hoom_ccan": "HOOM-CCAN",
}

def find_entry(experiments, base_name, phase):
    for e in reversed(experiments):
        name = e["exp_name"]
        parts = name.split("_", 2)
        base = parts[2] if len(parts) > 2 else name
        if "_test_" in base:
            base = base[:base.index("_test_")]
        if base == base_name and e["phase"] == phase:
            return e
    return None

entries = {}
for model_key, label in MODELS.items():
    train_e = find_entry(data["experiments"], model_key, "train")
    test_e = find_entry(data["experiments"], model_key, "test")
    if train_e and test_e:
        entries[label] = {"train": train_e["metrics"], "test": test_e["metrics"]}

labels = list(entries.keys())
out_dir = Path("experiments/plots")
out_dir.mkdir(exist_ok=True)

# --- 1. SELD score bar chart (train vs test) ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(labels))
w = 0.35
train_seld = [entries[l]["train"]["seld_scr"] for l in labels]
test_seld = [entries[l]["test"]["seld_scr"] for l in labels]

bars1 = ax.bar(x - w/2, train_seld, w, label="Train (val)")
bars2 = ax.bar(x + w/2, test_seld, w, label="Test")
ax.set_ylabel("SELD Score (lower is better)")
ax.set_title("SELD Score Comparison")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.bar_label(bars1, fmt="%.4f", fontsize=8)
ax.bar_label(bars2, fmt="%.4f", fontsize=8)
ax.set_ylim(0, max(train_seld + test_seld) * 1.15)
fig.tight_layout()
fig.savefig(out_dir / "seld_comparison.png", dpi=150)
print(f"Saved {out_dir / 'seld_comparison.png'}")
plt.close(fig)

# --- 2. Radar chart (test metrics, 5 axes) ---
# Normalize: ER↓, LE↓, SELD↓ (invert so bigger=better), F↑, LR↑ (keep)
metric_keys = ["ER", "F", "LE", "LR", "seld_scr"]
metric_labels = ["1-ER", "F-score", "1-LE/180", "LR", "1-SELD"]

def normalize_for_radar(m):
    return [
        1 - m["ER"],          # ER: lower is better → invert
        m["F"],               # F: higher is better
        1 - m["LE"] / 180,    # LE: lower is better → normalize & invert
        m["LR"],              # LR: higher is better
        1 - m["seld_scr"],    # SELD: lower is better → invert
    ]

angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
angles += angles[:1]  # close polygon

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))

for i, label in enumerate(labels):
    vals = normalize_for_radar(entries[label]["test"])
    vals += vals[:1]
    ax.plot(angles, vals, "o-", label=label, color=colors[i], linewidth=2, markersize=5)
    ax.fill(angles, vals, alpha=0.1, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1)
ax.set_title("Test Metrics Radar (outer = better)", y=1.08)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
fig.tight_layout()
fig.savefig(out_dir / "radar_comparison.png", dpi=150)
print(f"Saved {out_dir / 'radar_comparison.png'}")
plt.close(fig)

"""HOOM-fix vs HOOM-CCAN comparison chart with config annotations."""
import tomli
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

with open("experiments/metrics.toml", "rb") as f:
    data = tomli.load(f)

def find_entry(base_name, phase):
    for e in reversed(data["experiments"]):
        name = e["exp_name"]
        parts = name.split("_", 2)
        base = parts[2] if len(parts) > 2 else name
        if "_test_" in base:
            base = base[:base.index("_test_")]
        if base == base_name and e["phase"] == phase:
            return e
    return None

models = {
    "HOOM-fix\nccan_ch=[64,128]": find_entry("hoom_fix", "test")["metrics"],
    "HOOM-CCAN\nccan_ch=[64,64]": find_entry("hoom_ccan", "test")["metrics"],
}

out_dir = Path("experiments/plots")
out_dir.mkdir(exist_ok=True)

labels = list(models.keys())
metric_keys = ["ER", "F", "LE", "LR", "seld_scr"]
metric_display = ["ER ↓", "F ↑", "LE ↓", "LR ↑", "SELD ↓"]

# --- Bar chart: all 5 metrics side by side ---
fig, axes = plt.subplots(1, 5, figsize=(14, 5))
colors = ["#5B9BD5", "#ED7D31"]
x = np.arange(len(labels))

for i, (key, display) in enumerate(zip(metric_keys, metric_display)):
    ax = axes[i]
    vals = [models[l][key] for l in labels]
    bars = ax.bar(x, vals, color=colors, width=0.5)
    ax.set_title(display, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.bar_label(bars, fmt="%.4f", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.2)

fig.suptitle("HOOM Variants Test Comparison\nlayout=[ccan,ccan,buan,buan]  dim=256  conformer=8  mhsa=3",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(out_dir / "hoom_compare.png", dpi=150, bbox_inches="tight")
print(f"Saved {out_dir / 'hoom_compare.png'}")
plt.close(fig)

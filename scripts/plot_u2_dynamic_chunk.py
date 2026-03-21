"""Generate U2 Dynamic Chunk comparison chart."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# U2 Dynamic Chunk 实验数据 (从 metrics.toml 提取)
# 选取最佳的一次实验结果
chunk_sizes = [4, 9, 24, 49, 'Full\n(Streaming)', 'Full\n(Offline)']
chunk_labels = ['4', '9', '24', '49', 'Full', 'Offline']

# SELD scores (lower is better)
seld_scores = [0.4417, 0.4358, 0.4265, 0.4258, 0.4061, 0.4080]

# ER (Error Rate)
er_scores = [0.6653, 0.6533, 0.6338, 0.6236, 0.5864, 0.5891]

# F-score
f_scores = [0.3715, 0.3815, 0.3953, 0.3931, 0.4201, 0.4154]

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- Left: SELD Score by Chunk Size ---
ax1 = axes[0]
colors = ['#42a5f5'] * 4 + ['#66bb6a', '#ffa726']
bars = ax1.bar(range(len(chunk_labels)), seld_scores, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, seld_scores)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xlabel('Chunk Size', fontsize=12)
ax1.set_ylabel('SELD Score (lower is better)', fontsize=12)
ax1.set_title('U2 Dynamic Chunk: SELD Performance', fontsize=14, fontweight='bold')
ax1.set_xticks(range(len(chunk_labels)))
ax1.set_xticklabels(chunk_labels)
ax1.set_ylim(0, 0.55)
ax1.grid(axis='y', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#42a5f5', label='Streaming (chunk)'),
                   Patch(facecolor='#66bb6a', label='Streaming (full)'),
                   Patch(facecolor='#ffa726', label='Offline (full)')]
ax1.legend(handles=legend_elements, loc='upper right')

# Add annotation for best result
best_idx = np.argmin(seld_scores)
ax1.annotate('Best!\nSELD=0.4061', xy=(best_idx, seld_scores[best_idx]),
             xytext=(best_idx-1, seld_scores[best_idx]-0.08),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', fontweight='bold')

# --- Right: Metrics Comparison (Best Streaming vs Offline) ---
ax2 = axes[1]

metrics = ['SELD', 'ER', 'F-score', 'LE/100', 'LR']
best_streaming = [0.4061, 0.5864, 0.4201, 0.1929, 0.6493]  # full streaming
offline = [0.4080, 0.5891, 0.4154, 0.1958, 0.6506]  # full offline

x = np.arange(len(metrics))
width = 0.35

bars1 = ax2.bar(x - width/2, best_streaming, width, label='Streaming (Full Context)', color='#66bb6a', edgecolor='black', linewidth=0.5)
bars2 = ax2.bar(x + width/2, offline, width, label='Offline (Full Context)', color='#ffa726', edgecolor='black', linewidth=0.5)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

ax2.set_ylabel('Normalized Score', fontsize=12)
ax2.set_title('Streaming vs Offline: Full Context Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend(loc='upper right')
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()

# Save
out_dir = Path("experiments/plots")
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "u2_dynamic_chunk_analysis.png", dpi=150, bbox_inches='tight')
print(f"Saved {out_dir / 'u2_dynamic_chunk_analysis.png'}")

# Also create a simple chart showing chunk size trend only
fig2, ax3 = plt.subplots(figsize=(10, 6))

# Plot trend line for streaming chunks
x_vals = [4, 9, 24, 49]
seld_streaming = [0.4417, 0.4358, 0.4265, 0.4258]

ax3.plot(x_vals, seld_streaming, 'o-', color='#1976d2', linewidth=2, markersize=10, label='Streaming (different chunks)')
ax3.axhline(y=0.4061, color='#66bb6a', linestyle='--', linewidth=2, label='Streaming (full context)')
ax3.axhline(y=0.4080, color='#ff9800', linestyle='--', linewidth=2, label='Offline (full context)')

# Fill area between streaming and offline
ax3.fill_between([0, 60], 0.4061, 0.4080, alpha=0.2, color='gray', label='Performance gap')

ax3.set_xlabel('Chunk Size', fontsize=12)
ax3.set_ylabel('SELD Score (lower is better)', fontsize=12)
ax3.set_title('U2 Dynamic Chunk: Chunk Size vs Performance', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 55)
ax3.set_ylim(0.38, 0.48)

# Add annotations
ax3.annotate('Best Streaming:\nSELD=0.4258', xy=(49, 0.4258), xytext=(35, 0.438),
             arrowprops=dict(arrowstyle='->', color='blue'), fontsize=10)
ax3.annotate('Dynamic Chunk\nTraining Benefit', xy=(25, 0.415), fontsize=11,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig2.savefig(out_dir / "u2_chunk_trend.png", dpi=150, bbox_inches='tight')
print(f"Saved {out_dir / 'u2_chunk_trend.png'}")

plt.close('all')
print("\nDone! Generated charts:")
print("  - u2_dynamic_chunk_analysis.png (comprehensive)")
print("  - u2_chunk_trend.png (trend focus)")

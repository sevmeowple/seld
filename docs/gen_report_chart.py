import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

# --- Fig 1: Streaming vs Offline bar chart (1s segment) ---
configs = ['chunk_4\n(0.5s)', 'chunk_9\n(1.0s)', 'chunk_24\n(2.5s)', 'chunk_49\n(5.0s)', 'full_ctx\n(∞)']
streaming_seld = [0.4417, 0.4358, 0.4265, 0.4258, 0.4263]
offline_seld_1s = [None, None, None, None, 0.4702]
offline_seld_10s = [None, None, None, None, 0.4139]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(configs))
w = 0.28

bars1 = ax.bar(x - w, streaming_seld, w, label='Streaming (cache)', color='#4C72B0', zorder=3)
# offline only has full_ctx
offline_vals = [0] * 4 + [0.4702]
bars2 = ax.bar([4], [0.4702], w, label='Offline (1s seg)', color='#DD8452', zorder=3)
bars3 = ax.bar([4 + w], [0.4139], w, label='Offline (10s seg)', color='#55A868', zorder=3)

ax.set_ylabel('SELD Score ↓')
ax.set_title('U2 Dynamic Chunk: Single Model, Multiple Inference Modes')
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylim(0.38, 0.50)
ax.axhline(y=0.40, color='red', linestyle='--', alpha=0.7, label='Target (0.40)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3, zorder=0)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)
ax.text(4, 0.4702 + 0.002, '0.4702', ha='center', va='bottom', fontsize=8)
ax.text(4 + w, 0.4139 + 0.002, '0.4139', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('docs/fig1_seld_comparison.png', dpi=150)
plt.close()

# --- Fig 2: Architecture diagram (text-based) ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('ResNet-Conformer U2 Architecture: Dual Inference Path', fontsize=13, fontweight='bold', pad=15)

# Shared backbone
boxes = [
    (1.5, 8.5, 3.0, 0.8, '#B0C4DE', 'Input: FOA 4ch Audio'),
    (1.5, 7.2, 3.0, 0.8, '#87CEEB', 'ResNet18 (no pool)'),
    (1.5, 5.9, 3.0, 0.8, '#87CEEB', 'Linear Projection'),
    (1.5, 4.6, 3.0, 0.8, '#87CEEB', 'Conformer ×8'),
    (1.5, 3.3, 3.0, 0.8, '#87CEEB', 'MaxPool1d(5)'),
    (0.5, 1.8, 2.0, 0.8, '#98FB98', 'SED (σ)'),
    (3.5, 1.8, 2.0, 0.8, '#98FB98', 'DOA (tanh)'),
]
for x0, y0, w, h, color, text in boxes:
    ax.add_patch(mpatches.Rectangle((x0, y0), w, h, facecolor=color, edgecolor='black', linewidth=1.2, zorder=2))
    ax.text(x0 + w/2, y0 + h/2, text, ha='center', va='center', fontsize=9, zorder=3)

# Arrows for shared path
for i in range(4):
    y_from = boxes[i][1]
    y_to = boxes[i+1][1] + boxes[i+1][3]
    ax.annotate('', xy=(3.0, y_to), xytext=(3.0, y_from),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
# MaxPool to heads
ax.annotate('', xy=(1.5, 2.6), xytext=(2.5, 3.3), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.annotate('', xy=(4.5, 2.6), xytext=(3.5, 3.3), arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Right side: mode explanation
mode_x = 6.2
ax.text(mode_x, 9.2, 'Inference Mode Control', fontsize=11, fontweight='bold')

# Offline box
ax.add_patch(mpatches.Rectangle((mode_x, 7.5), 3.5, 1.4, facecolor='#FFF8DC', edgecolor='#DD8452', linewidth=2, zorder=2))
ax.text(mode_x + 0.15, 8.55, 'Offline Path', fontsize=10, fontweight='bold', color='#DD8452')
ax.text(mode_x + 0.15, 8.1, 'model(x)', fontsize=9, family='monospace')
ax.text(mode_x + 0.15, 7.7, 'Chunk mask on attention\nFull sequence in one pass', fontsize=8)

# Streaming box
ax.add_patch(mpatches.Rectangle((mode_x, 5.5), 3.5, 1.7, facecolor='#F0F8FF', edgecolor='#4C72B0', linewidth=2, zorder=2))
ax.text(mode_x + 0.15, 6.85, 'Streaming Path', fontsize=10, fontweight='bold', color='#4C72B0')
ax.text(mode_x + 0.15, 6.4, 'model(x, resnet_cache,\n      conformer_cache)', fontsize=9, family='monospace')
ax.text(mode_x + 0.15, 5.7, 'Cache-based incremental inference\nQ=current, K/V=cache+current', fontsize=8)

# Training box
ax.add_patch(mpatches.Rectangle((mode_x, 3.5), 3.5, 1.7, facecolor='#F5F5DC', edgecolor='#55A868', linewidth=2, zorder=2))
ax.text(mode_x + 0.15, 4.85, 'U2 Training Strategy', fontsize=10, fontweight='bold', color='#55A868')
ax.text(mode_x + 0.15, 4.35, '~50% full context (offline)\n~50% random chunk (streaming)', fontsize=9)
ax.text(mode_x + 0.15, 3.7, 'Single model learns both paths\nvia dynamic chunk masking', fontsize=8)

plt.tight_layout()
plt.savefig('docs/fig2_architecture.png', dpi=150)
plt.close()

print("Charts saved to docs/")

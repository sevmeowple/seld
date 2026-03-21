"""Plot ResNet-Conformer-DHOOM architecture diagram."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, ConnectionPatch
import numpy as np

# Set up the figure with warm beige background like the reference
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#FAF3E8')  # Warm beige background
ax.set_facecolor('#FAF3E8')

# Color palette (matching academic style)
COLORS = {
    'input': '#F4A460',      # Sandy brown for input
    'resnet_offline': '#B0C4DE',  # Light steel blue
    'resnet_online': '#FFB6C1',   # Light pink
    'conformer_offline': '#ADD8E6',  # Light blue
    'conformer_online': '#FFDAB9',   # Peach puff
    'fusion': '#DDA0DD',     # Plum for fusion
    'output': '#98FB98',     # Pale green
    'mhsa': '#F0E68C',       # Khaki
    'text': '#333333',
    'arrow': '#888888',
}

def draw_box(ax, x, y, width, height, color, text, text_size=8, border_color=None, radius=0.05):
    """Draw a rounded rectangle box."""
    if border_color is None:
        border_color = color
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle=f"round,pad=0.02,rounding_size={radius}",
                         facecolor=color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=text_size,
            color=COLORS['text'], weight='bold', wrap=True)
    return box

def draw_arrow(ax, start, end, color='#888888', style='->', lw=1.5):
    """Draw an arrow."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

# ==================== INPUT SECTION (Left) ====================
# Audio Input
ax.text(0.8, 9.5, 'Audio Input', fontsize=10, weight='bold', ha='center')
draw_box(ax, 0.8, 8.8, 1.4, 0.5, COLORS['input'], '7-ch FOA\n(4 Mel + 3 IV)', 7)

# Waveform icon
for i in range(3):
    y_offset = 9.2 + i * 0.15
    x_wave = np.linspace(0.4, 1.2, 20)
    y_wave = y_offset + 0.05 * np.sin(x_wave * 20 + i)
    ax.plot(x_wave, y_wave, 'b-', linewidth=0.8, alpha=0.6)

# Arrow to ResNet
draw_arrow(ax, (1.5, 8.8), (2.3, 8.8))

# ==================== RESNET SECTION ====================
# Title
ax.text(4.5, 9.5, 'Dual-Path ResNet18 (8× BasicBlock Interaction)',
        fontsize=10, weight='bold', ha='center')

# Offline ResNet (Top)
ax.text(3.0, 9.0, 'Offline Path\n(Non-causal)', fontsize=8, ha='center',
        color='#4169E1', weight='bold')

# Online ResNet (Bottom)
ax.text(3.0, 7.0, 'Online Path\n(Causal)', fontsize=8, ha='center',
        color='#DC143C', weight='bold')

# ResNet Blocks - showing the interaction pattern
resnet_blocks = [
    ('Conv1\n+ LN', 8.5),
    ('Layer1\n(2 blocks)', 7.8),
    ('MaxPool', 7.2),
    ('Layer2\n(2 blocks)', 6.4),
    ('MaxPool', 5.8),
    ('Layer3\n(2 blocks)', 5.0),
    ('MaxPool', 4.4),
    ('Layer4\n(2 blocks)', 3.6),
    ('Conv5\n1×1', 2.9),
]

# Draw Offline ResNet blocks
offline_x = 4.5
for text, y in resnet_blocks:
    draw_box(ax, offline_x, y, 1.4, 0.5, COLORS['resnet_offline'], text, 7)

# Draw Online ResNet blocks
online_x = 4.5
for text, y in resnet_blocks:
    draw_box(ax, online_x, y - 2.0, 1.4, 0.5, COLORS['resnet_online'], text, 7)

# Draw fusion arrows (bidirectional)
fusion_y_pairs = [
    (8.5, 6.5),   # Conv1
    (7.8, 5.8),   # Layer1
    (6.4, 4.4),   # Layer2
    (5.0, 3.0),   # Layer3
    (3.6, 1.6),   # Layer4
]

for off_y, on_y in fusion_y_pairs:
    # Arrow from offline to online
    ax.annotate('', xy=(4.5, on_y + 0.25), xytext=(4.5, off_y - 0.25),
                arrowprops=dict(arrowstyle='->', color='#9370DB', lw=1.5,
                              connectionstyle="arc3,rad=0.3"))
    # Arrow from online to offline
    ax.annotate('', xy=(4.5, off_y - 0.25), xytext=(4.5, on_y + 0.25),
                arrowprops=dict(arrowstyle='->', color='#9370DB', lw=1.5,
                              connectionstyle="arc3,rad=-0.3"))

# Fusion labels
ax.text(5.6, 7.5, 'Cross-Path\nFusion', fontsize=7, ha='center',
        color='#9370DB', style='italic')

# ResNet output projection
ax.text(4.5, 1.5, 'Flatten + Linear', fontsize=7, ha='center', style='italic')
draw_arrow(ax, (4.5, 2.6), (4.5, 2.0))
draw_arrow(ax, (4.5, 0.65), (4.5, 0.2))

# ==================== CONFORMER SECTION ====================
ax.text(8.5, 9.5, 'Dual-Path Conformer (8 Layers)',
        fontsize=10, weight='bold', ha='center')

# Conformer blocks
conformer_y_offline = 8.0
conformer_y_online = 4.0

# Draw stacked conformer layers representation
for i in range(4):  # Show 4 blocks as representation
    y_off = conformer_y_offline - i * 0.8
    y_on = conformer_y_online - i * 0.8

    # Offline layer
    draw_box(ax, 8.5, y_off, 1.6, 0.6, COLORS['conformer_offline'],
             f'Offline Layer\n{i*2+1}-{i*2+2}', 7)

    # Online layer
    draw_box(ax, 8.5, y_on, 1.6, 0.6, COLORS['conformer_online'],
             f'Online Layer\n{i*2+1}-{i*2+2}', 7)

    # Fusion arrows between them
    if i < 3:
        ax.annotate('', xy=(8.5, y_on + 0.5), xytext=(8.5, y_off - 0.4),
                    arrowprops=dict(arrowstyle='->', color='#9370DB', lw=1.2))
        ax.annotate('', xy=(8.5, y_off - 0.4), xytext=(8.5, y_on + 0.5),
                    arrowprops=dict(arrowstyle='->', color='#9370DB', lw=1.2))

# Arrow from ResNet to Conformer
draw_arrow(ax, (5.2, 0.8), (7.3, 4.5))  # To online
draw_arrow(ax, (5.2, 0.8), (7.3, 5.5))  # To offline

# ==================== MHSA & OUTPUT SECTION ====================
# MHSA for offline path
draw_box(ax, 10.5, 7.0, 1.2, 0.5, COLORS['mhsa'], 'MHSA\n(optional)', 7)
draw_arrow(ax, (9.3, 5.2), (9.9, 6.7))

# Output heads
ax.text(10.5, 9.0, 'Output Heads', fontsize=10, weight='bold', ha='center')

# Offline output
draw_box(ax, 10.5, 5.5, 1.4, 0.6, COLORS['output'], 'Offline Head\nSED (sigmoid)\nDOA (tanh)', 7)
draw_arrow(ax, (10.5, 6.7), (10.5, 5.9))

# Online output
draw_box(ax, 10.5, 2.5, 1.4, 0.6, COLORS['output'], 'Online Head\nSED (sigmoid)\nDOA (tanh)', 7)
draw_arrow(ax, (9.3, 3.2), (9.9, 2.5))

# ==================== RIGHT SIDE: OUTPUT VISUALIZATION ====================
# DOA output visualization
ax.text(12.5, 8.5, 'DOA Output', fontsize=10, weight='bold', ha='center')
ax.text(12.5, 8.1, '(x, y, z) per frame', fontsize=8, ha='center', style='italic')

# Simple 3D coordinate visualization
coord_box = FancyBboxPatch((11.8, 6.8), 1.4, 1.0,
                           boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor='#333', linewidth=1)
ax.add_patch(coord_box)
ax.text(12.5, 7.7, 'frame t', fontsize=7, ha='center')
ax.plot([12.0, 12.3], [7.2, 7.2], 'r-', linewidth=2)
ax.plot([12.0, 12.0], [7.2, 7.5], 'g-', linewidth=2)
ax.text(12.15, 7.0, 'x', fontsize=7, color='red')
ax.text(11.85, 7.4, 'y', fontsize=7, color='green')

# SED output visualization
ax.text(12.5, 5.5, 'SED Output', fontsize=10, weight='bold', ha='center')
ax.text(12.5, 5.1, 'Event classes', fontsize=8, ha='center', style='italic')

# Event timeline
sed_box = FancyBboxPatch((11.8, 3.8), 1.4, 1.0,
                         boxstyle="round,pad=0.02",
                         facecolor='white', edgecolor='#333', linewidth=1)
ax.add_patch(sed_box)
ax.text(12.5, 4.6, 'frame t', fontsize=7, ha='center')
# Event bars
events = [('Music', 4.3, 'skyblue'), ('Walk', 4.15, 'lightgreen'), ('Knock', 4.0, 'lightyellow')]
for name, y, color in events:
    rect = Rectangle((12.0, y), 0.5, 0.1, facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    ax.text(11.9, y+0.05, name, fontsize=6, ha='right', va='center')

# ==================== LEGEND ====================
legend_items = [
    (COLORS['resnet_offline'], 'Offline ResNet'),
    (COLORS['resnet_online'], 'Online ResNet'),
    (COLORS['conformer_offline'], 'Offline Conformer'),
    (COLORS['conformer_online'], 'Online Conformer'),
    ('#9370DB', 'Cross-Path Fusion'),
]

ax.text(0.5, 3.0, 'Legend:', fontsize=9, weight='bold')
for i, (color, label) in enumerate(legend_items):
    y = 2.5 - i * 0.35
    rect = Rectangle((0.3, y), 0.3, 0.2, facecolor=color, edgecolor='black', linewidth=0.5)
    ax.add_patch(rect)
    ax.text(0.7, y + 0.1, label, fontsize=8, va='center')

# Key features box
features_text = (
    "Key Features:\n"
    "• 8× ResNet BasicBlock interactions\n"
    "• 8× Conformer layer-wise fusion\n"
    "• Dual inference: offline + streaming\n"
    "• Unified training with cache support"
)
ax.text(0.5, 0.8, features_text, fontsize=8, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

# Title
ax.text(7, 9.9, 'ResNet-Conformer-DHOOM Architecture',
        fontsize=14, weight='bold', ha='center')
ax.text(7, 9.6, 'Dual-path Hybrid Offline-Online Model with Progressive Layer-wise Fusion',
        fontsize=10, ha='center', style='italic')

plt.tight_layout()
plt.savefig('/disk7/zchan/Proj/SELD/docs/resnet_conformer_dhoom_architecture.png',
            dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.savefig('/disk7/zchan/Proj/SELD/docs/resnet_conformer_dhoom_architecture.pdf',
            bbox_inches='tight', facecolor=fig.get_facecolor())
print("Architecture diagram saved!")
plt.show()

# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import numpy as np

# 添加自定义字体
font_path = '/disk7/zchan/bin/font/LXGWWenKai-Medium.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'LXGW WenKai'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False

# --- Fig 1: All results comparison ---
# Group 1: streaming 1s segment (different chunk sizes)
# Group 2: 10s segment (streaming + offline)
# Group 3: 1s segment offline (for reference)

labels = ['chunk_4\n(0.1s)', 'chunk_9\n(0.2s)', 'chunk_24\n(0.5s)', 'chunk_49\n(1.0s)',
          '全上下文\n流式 1s', '全上下文\n离线 1s',
          '全上下文\n流式 10s', '全上下文\n离线 10s']
values = [0.4417, 0.4358, 0.4265, 0.4258, 0.4263, 0.4702, 0.4061, 0.4139]
colors = ['#4C72B0']*5 + ['#DD8452'] + ['#4C72B0', '#DD8452']

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels))
bars = ax.bar(x, values, 0.6, color=colors, zorder=3, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('SELD 分数 ↓')
ax.set_title('U2 动态块训练：单模型多推理模式', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.38, 0.51)
ax.axhline(y=0.40, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='目标 (0.40)')

# Divider lines
ax.axvline(x=4.5, color='gray', linestyle=':', alpha=0.5)
ax.axvline(x=5.5, color='gray', linestyle=':', alpha=0.5)

# Region labels
ax.text(2, 0.495, '流式推理 (1s 分段测试)', ha='center', fontsize=9, color='gray', style='italic')
ax.text(7, 0.495, '10s 分段测试', ha='center', fontsize=9, color='gray', style='italic')

# Legend
stream_patch = mpatches.Patch(color='#4C72B0', label='流式 (使用 cache)')
offline_patch = mpatches.Patch(color='#DD8452', label='离线 (无 cache)')
ax.legend(handles=[stream_patch, offline_patch], loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig('docs/fig1_seld_comparison.png', dpi=150)
plt.close()
print("Fig1 saved")

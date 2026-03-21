# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# 添加自定义字体
font_path = '/disk7/zchan/bin/font/LXGWWenKai-Medium.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'LXGW WenKai'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

# U2 离线数据
labels = [
    '1s',
    '2s',
    '2s\noverlap',
    '5s',
    '5s\noverlap',
    '7s',
    '7s\noverlap',
    '10s',
    '10s\noverlap'
]

values = [
    0.470207,  # U2 1s
    0.440684,  # U2 2s
    0.432566,  # U2 2s overlap
    0.421616,  # U2 5s
    0.417464,  # U2 5s overlap
    0.414333,  # U2 7s
    0.410473,  # U2 7s overlap
    0.41392,   # U2 10s
    0.407964   # U2 10s overlap
]

colors = ['#4C72B0'] * 9

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(labels))
bars = ax.bar(x, values, 0.6, color=colors, zorder=3, edgecolor='white', linewidth=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('SELD 分数 ↓')
ax.set_title('U2 离线模式性能对比', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylim(0.38, 0.50)
ax.axhline(y=0.40, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
ax.grid(axis='y', alpha=0.3, zorder=0)

plt.tight_layout()
plt.savefig('docs/offline_comparison.png', dpi=150)
plt.close()
print("U2 offline comparison chart saved")

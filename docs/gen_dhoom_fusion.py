# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.font_manager as fm

font_path = '/disk7/zchan/bin/font/LXGWWenKai-Medium.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'LXGW WenKai'
plt.rcParams['font.size'] = 36
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

# 标题
ax.text(6, 9.3, '单层融合过程', ha='center', fontsize=48, fontweight='bold')

# 输入特征 - 增大间距
box1 = FancyBboxPatch((2.0, 8.0), 2.0, 0.7, boxstyle="round,pad=0.05",
                      edgecolor='#DD8452', facecolor='#fff3e6', linewidth=4)
ax.add_patch(box1)
ax.text(3.0, 8.35, '离线特征 i', ha='center', va='center', fontsize=34, fontweight='bold')

box2 = FancyBboxPatch((8.0, 8.0), 2.0, 0.7, boxstyle="round,pad=0.05",
                      edgecolor='#4C72B0', facecolor='#e3f2fd', linewidth=4)
ax.add_patch(box2)
ax.text(9.0, 8.35, '在线特征 i', ha='center', va='center', fontsize=34, fontweight='bold')

# 交叉箭头
ax.arrow(3.0, 8.0, 5.5, -0.8, head_width=0.2, head_length=0.15,
         fc='#666', ec='#666', lw=4, linestyle='--', alpha=0.8)
ax.arrow(9.0, 8.0, -5.5, -0.8, head_width=0.2, head_length=0.15,
         fc='#666', ec='#666', lw=4, linestyle='--', alpha=0.8)

# 融合后
box3 = FancyBboxPatch((1.5, 6.3), 3.0, 0.8, boxstyle="round,pad=0.05",
                      edgecolor='#DD8452', facecolor='#fff3e6', linewidth=4)
ax.add_patch(box3)
ax.text(3.0, 6.7, '离线 + 在线', ha='center', va='center', fontsize=34, fontweight='bold')

box4 = FancyBboxPatch((7.5, 6.3), 3.0, 0.8, boxstyle="round,pad=0.05",
                      edgecolor='#4C72B0', facecolor='#e3f2fd', linewidth=4)
ax.add_patch(box4)
ax.text(9.0, 6.7, '在线 + 离线', ha='center', va='center', fontsize=34, fontweight='bold')

# 下箭头
arrow1 = FancyArrowPatch((3.0, 6.3), (3.0, 5.3), arrowstyle='->', lw=5, color='#DD8452')
ax.add_patch(arrow1)
arrow2 = FancyArrowPatch((9.0, 6.3), (9.0, 5.3), arrowstyle='->', lw=5, color='#4C72B0')
ax.add_patch(arrow2)

# 网络层
box5 = FancyBboxPatch((1.5, 3.8), 3.0, 1.4, boxstyle="round,pad=0.05",
                      edgecolor='#DD8452', facecolor='#fff3e6', linewidth=4)
ax.add_patch(box5)
ax.text(3.0, 4.5, '网络层 i', ha='center', va='center', fontsize=36, fontweight='bold')

box6 = FancyBboxPatch((7.5, 3.8), 3.0, 1.4, boxstyle="round,pad=0.05",
                      edgecolor='#4C72B0', facecolor='#e3f2fd', linewidth=4)
ax.add_patch(box6)
ax.text(9.0, 4.5, '网络层 i', ha='center', va='center', fontsize=36, fontweight='bold')

# 输出
arrow3 = FancyArrowPatch((3.0, 3.8), (3.0, 2.8), arrowstyle='->', lw=5, color='#DD8452')
ax.add_patch(arrow3)
arrow4 = FancyArrowPatch((9.0, 3.8), (9.0, 2.8), arrowstyle='->', lw=5, color='#4C72B0')
ax.add_patch(arrow4)

box7 = FancyBboxPatch((2.0, 1.8), 2.0, 0.8, boxstyle="round,pad=0.05",
                      edgecolor='#DD8452', facecolor='#fff3e6', linewidth=4)
ax.add_patch(box7)
ax.text(3.0, 2.2, '离线特征 i+1', ha='center', va='center', fontsize=32, fontweight='bold')

box8 = FancyBboxPatch((8.0, 1.8), 2.0, 0.8, boxstyle="round,pad=0.05",
                      edgecolor='#4C72B0', facecolor='#e3f2fd', linewidth=4)
ax.add_patch(box8)
ax.text(9.0, 2.2, '在线特征 i+1', ha='center', va='center', fontsize=32, fontweight='bold')

plt.tight_layout()
plt.savefig('docs/dhoom_fusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("DHOOM fusion mechanism diagram saved to docs/dhoom_fusion.png")

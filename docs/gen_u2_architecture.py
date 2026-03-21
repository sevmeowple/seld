# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 添加自定义字体
font_path = '/disk7/zchan/bin/font/LXGWWenKai-Medium.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'LXGW WenKai'
plt.rcParams['font.size'] = 36  # 24 * 1.5 = 36
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(20, 8))  # 继续放大画布
ax.set_xlim(0, 20)
ax.set_ylim(0, 8)
ax.axis('off')

# 标题
ax.text(10, 7.5, 'U2 Dynamic Chunk：单模型支持流式/离线双模式推理',
        ha='center', fontsize=42, fontweight='bold')

# ========== 左列：训练流程 ==========
ax.text(3.5, 6.8, '训练流程', ha='center', fontsize=36, fontweight='bold', color='#0066cc')

# 输入
box1 = FancyBboxPatch((2.3, 6.1), 2.4, 0.5, boxstyle="round,pad=0.05",
                       edgecolor='black', facecolor='#e3f2fd', linewidth=3)
ax.add_patch(box1)
ax.text(3.5, 6.35, '音频输入', ha='center', va='center', fontsize=30)

# ResNet
arrow1 = FancyArrowPatch((3.5, 6.1), (3.5, 5.6), arrowstyle='->', lw=3, color='black')
ax.add_patch(arrow1)
box2 = FancyBboxPatch((2.3, 4.9), 2.4, 0.6, boxstyle="round,pad=0.05",
                       edgecolor='black', facecolor='#bbdefb', linewidth=3)
ax.add_patch(box2)
ax.text(3.5, 5.2, 'ResNet18', ha='center', va='center', fontsize=30, fontweight='bold')

# Projection
arrow2 = FancyArrowPatch((3.5, 4.9), (3.5, 4.4), arrowstyle='->', lw=3, color='black')
ax.add_patch(arrow2)
box3 = FancyBboxPatch((2.3, 3.9), 2.4, 0.45, boxstyle="round,pad=0.05",
                       edgecolor='black', facecolor='#90caf9', linewidth=3)
ax.add_patch(box3)
ax.text(3.5, 4.125, 'Projection', ha='center', va='center', fontsize=27)

# Conformer
arrow3 = FancyArrowPatch((3.5, 3.9), (3.5, 3.4), arrowstyle='->', lw=3, color='black')
ax.add_patch(arrow3)
box4 = FancyBboxPatch((1.8, 2.2), 3.4, 1.1, boxstyle="round,pad=0.05",
                       edgecolor='black', facecolor='#64b5f6', linewidth=3)
ax.add_patch(box4)
ax.text(3.5, 3.0, 'Conformer (×8)', ha='center', va='center', fontsize=30, fontweight='bold')
ax.text(3.5, 2.55, '动态 Mask', ha='center', va='center', fontsize=27, style='italic')

# 输出
arrow4 = FancyArrowPatch((3.5, 2.3), (3.5, 1.8), arrowstyle='->', lw=3, color='black')
ax.add_patch(arrow4)
box5 = FancyBboxPatch((2.3, 1.1), 2.4, 0.6, boxstyle="round,pad=0.05",
                       edgecolor='black', facecolor='#42a5f5', linewidth=3)
ax.add_patch(box5)
ax.text(3.5, 1.4, 'SED + DOA', ha='center', va='center', fontsize=30, fontweight='bold')

# ========== 中列：动态 Mask 机制 ==========
ax.text(10, 6.8, '动态 Mask 机制', ha='center', fontsize=36, fontweight='bold', color='#0066cc')

mask_box = FancyBboxPatch((7.5, 5.0), 5, 1.5, boxstyle="round,pad=0.1",
                           edgecolor='#666', facecolor='#fff9c4', linewidth=3)
ax.add_patch(mask_box)
ax.text(10, 6.1, '训练时随机采样', ha='center', fontsize=30, fontweight='bold')
ax.text(10, 5.6, '块大小: 1~50 frames', ha='center', fontsize=28)
ax.text(10, 5.15, '50% 完整 + 50% 随机', ha='center', fontsize=28)

# 推理模式
ax.text(10, 4.3, '推理模式', ha='center', fontsize=36, fontweight='bold', color='#0066cc')

box_offline = FancyBboxPatch((7.5, 3.0), 5, 1.0, boxstyle="round,pad=0.1",
                              edgecolor='#DD8452', facecolor='#fff3e6', linewidth=3.5)
ax.add_patch(box_offline)
ax.text(10, 3.7, '离线推理', ha='center', fontsize=32, fontweight='bold', color='#DD8452')
ax.text(10, 3.25, '完整上下文 | 无 cache', ha='center', fontsize=27)

box_stream = FancyBboxPatch((7.5, 1.5), 5, 1.0, boxstyle="round,pad=0.1",
                             edgecolor='#4C72B0', facecolor='#e3f2fd', linewidth=3.5)
ax.add_patch(box_stream)
ax.text(10, 2.2, '流式推理', ha='center', fontsize=32, fontweight='bold', color='#4C72B0')
ax.text(10, 1.75, '固定块 | 使用 cache', ha='center', fontsize=27)

# ========== 右列：核心优势 ==========
ax.text(16.5, 6.8, '核心优势', ha='center', fontsize=36, fontweight='bold', color='#2e7d32')

key_box = FancyBboxPatch((14, 1.5), 5, 5.1, boxstyle="round,pad=0.1",
                          edgecolor='#2e7d32', facecolor='#e8f5e9', linewidth=3.5)
ax.add_patch(key_box)

ax.text(16.5, 5.9, '✓ 单模型', ha='center', fontsize=30, fontweight='bold')
ax.text(16.5, 5.4, '支持多种推理配置', ha='center', fontsize=28)

ax.text(16.5, 4.6, '✓ 训练时学习', ha='center', fontsize=30, fontweight='bold')
ax.text(16.5, 4.1, '不同上下文长度', ha='center', fontsize=28)

ax.text(16.5, 3.3, '✓ 推理时', ha='center', fontsize=30, fontweight='bold')
ax.text(16.5, 2.8, '灵活切换模式', ha='center', fontsize=28)

ax.text(16.5, 2.0, '✓ 无需分别训练', ha='center', fontsize=30, fontweight='bold')
ax.text(16.5, 1.5, '流式/离线模型', ha='center', fontsize=28)

plt.tight_layout()
plt.savefig('docs/u2_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("U2 architecture diagram saved")

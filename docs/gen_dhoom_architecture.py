# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 添加自定义字体
font_path = '/disk7/zchan/bin/font/LXGWWenKai-Medium.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'LXGW WenKai'
plt.rcParams['font.size'] = 36
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

# 标题
ax.text(8, 7.5, 'DHOOM：双路径逐层融合架构',
        ha='center', fontsize=42, fontweight='bold')

# ========== 左列：双路径流程 ==========
ax.text(3.5, 6.8, '双路径流程', ha='center', fontsize=36, fontweight='bold', color='#0066cc')

# 输入
box_input = FancyBboxPatch((2.3, 6.1), 2.4, 0.5, boxstyle="round,pad=0.05",
                           edgecolor='black', facecolor='#e3f2fd', linewidth=3)
ax.add_patch(box_input)
ax.text(3.5, 6.35, '音频输入', ha='center', va='center', fontsize=30)

# 分叉箭头（增大分叉角度）
ax.arrow(3.5, 6.1, -0.9, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black', lw=2)
ax.arrow(3.5, 6.1, 0.9, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black', lw=2)

# ResNet 双路径（增大间距）
box_resnet_off = FancyBboxPatch((1.2, 4.9), 1.4, 0.6, boxstyle="round,pad=0.05",
                                edgecolor='#DD8452', facecolor='#fff3e6', linewidth=3)
ax.add_patch(box_resnet_off)
ax.text(1.9, 5.2, 'ResNet', ha='center', va='center', fontsize=26, fontweight='bold', color='#DD8452')

box_resnet_on = FancyBboxPatch((4.4, 4.9), 1.4, 0.6, boxstyle="round,pad=0.05",
                               edgecolor='#4C72B0', facecolor='#e3f2fd', linewidth=3)
ax.add_patch(box_resnet_on)
ax.text(5.1, 5.2, 'ResNet', ha='center', va='center', fontsize=26, fontweight='bold', color='#4C72B0')

# 融合标记（居中）
ax.text(3.5, 5.2, '⊕', ha='center', va='center', fontsize=40, color='#f57c00', fontweight='bold',
        bbox=dict(boxstyle='circle,pad=0.1', facecolor='#fff9c4', edgecolor='#f57c00', linewidth=2))
ax.text(3.5, 4.6, '×8', ha='center', fontsize=22, style='italic')

# Projection
arrow_p1 = FancyArrowPatch((1.9, 4.9), (1.9, 4.4), arrowstyle='->', lw=2, color='#DD8452')
ax.add_patch(arrow_p1)
arrow_p2 = FancyArrowPatch((5.1, 4.9), (5.1, 4.4), arrowstyle='->', lw=2, color='#4C72B0')
ax.add_patch(arrow_p2)

box_proj_off = FancyBboxPatch((1.2, 3.9), 1.4, 0.45, boxstyle="round,pad=0.05",
                              edgecolor='#DD8452', facecolor='#fff3e6', linewidth=2.5)
ax.add_patch(box_proj_off)
ax.text(1.9, 4.125, 'Projection', ha='center', va='center', fontsize=22)

box_proj_on = FancyBboxPatch((4.4, 3.9), 1.4, 0.45, boxstyle="round,pad=0.05",
                             edgecolor='#4C72B0', facecolor='#e3f2fd', linewidth=2.5)
ax.add_patch(box_proj_on)
ax.text(5.1, 4.125, 'Projection', ha='center', va='center', fontsize=22)

# Conformer 双路径
arrow_c1 = FancyArrowPatch((1.9, 3.9), (1.9, 3.4), arrowstyle='->', lw=2, color='#DD8452')
ax.add_patch(arrow_c1)
arrow_c2 = FancyArrowPatch((5.1, 3.9), (5.1, 3.4), arrowstyle='->', lw=2, color='#4C72B0')
ax.add_patch(arrow_c2)

box_conf_off = FancyBboxPatch((0.9, 2.2), 2.0, 1.1, boxstyle="round,pad=0.05",
                              edgecolor='#DD8452', facecolor='#fff3e6', linewidth=3)
ax.add_patch(box_conf_off)
ax.text(1.9, 3.0, 'Conformer', ha='center', va='center', fontsize=26, fontweight='bold')
ax.text(1.9, 2.55, '(×8)', ha='center', va='center', fontsize=24)

box_conf_on = FancyBboxPatch((4.1, 2.2), 2.0, 1.1, boxstyle="round,pad=0.05",
                             edgecolor='#4C72B0', facecolor='#e3f2fd', linewidth=3)
ax.add_patch(box_conf_on)
ax.text(5.1, 3.0, 'Conformer', ha='center', va='center', fontsize=26, fontweight='bold')
ax.text(5.1, 2.55, '(×8)', ha='center', va='center', fontsize=24)

# 融合标记
ax.text(3.5, 2.8, '⊕', ha='center', va='center', fontsize=40, color='#f57c00', fontweight='bold',
        bbox=dict(boxstyle='circle,pad=0.1', facecolor='#fff9c4', edgecolor='#f57c00', linewidth=2))
ax.text(3.5, 2.2, '×8', ha='center', fontsize=22, style='italic')

# 输出
arrow_o1 = FancyArrowPatch((1.9, 2.3), (1.9, 1.8), arrowstyle='->', lw=2, color='#DD8452')
ax.add_patch(arrow_o1)
arrow_o2 = FancyArrowPatch((5.1, 2.3), (5.1, 1.8), arrowstyle='->', lw=2, color='#4C72B0')
ax.add_patch(arrow_o2)

box_out_off = FancyBboxPatch((1.2, 1.1), 1.4, 0.6, boxstyle="round,pad=0.05",
                             edgecolor='#DD8452', facecolor='#fff3e6', linewidth=3)
ax.add_patch(box_out_off)
ax.text(1.9, 1.4, '离线输出', ha='center', va='center', fontsize=24, fontweight='bold')

box_out_on = FancyBboxPatch((4.4, 1.1), 1.4, 0.6, boxstyle="round,pad=0.05",
                            edgecolor='#4C72B0', facecolor='#e3f2fd', linewidth=3)
ax.add_patch(box_out_on)
ax.text(5.1, 1.4, '在线输出', ha='center', va='center', fontsize=24, fontweight='bold')

# ========== 右列：融合机制 ==========
ax.text(11.5, 6.8, '逐层融合机制', ha='center', fontsize=36, fontweight='bold', color='#0066cc')

# ResNet 融合
resnet_box = FancyBboxPatch((8.3, 5.3), 7.0, 1.2, boxstyle="round,pad=0.1",
                            edgecolor='#666', facecolor='#fff9c4', linewidth=3)
ax.add_patch(resnet_box)
ax.text(11.8, 6.15, 'ResNet BasicBlock 级', ha='center', fontsize=30, fontweight='bold')
ax.text(11.8, 5.7, 'offline = layer(offline + online)', ha='center', fontsize=24, family='monospace')
ax.text(11.8, 5.35, '8 个融合点（4层×2块）', ha='center', fontsize=24, style='italic')

# Conformer 融合
conf_box = FancyBboxPatch((8.3, 3.7), 7.0, 1.2, boxstyle="round,pad=0.1",
                          edgecolor='#666', facecolor='#fff9c4', linewidth=3)
ax.add_patch(conf_box)
ax.text(11.8, 4.55, 'Conformer 层级', ha='center', fontsize=30, fontweight='bold')
ax.text(11.8, 4.1, 'offline = conf[i](offline + online)', ha='center', fontsize=24, family='monospace')
ax.text(11.8, 3.75, '8 个融合点（每层融合）', ha='center', fontsize=24, style='italic')

# 训练/推理模式
ax.text(11.8, 3.0, '训练 vs 推理', ha='center', fontsize=36, fontweight='bold', color='#0066cc')

train_box = FancyBboxPatch((8.3, 2.0), 7.0, 0.8, boxstyle="round,pad=0.1",
                           edgecolor='#2e7d32', facecolor='#e8f5e9', linewidth=3)
ax.add_patch(train_box)
ax.text(11.8, 2.6, '训练：双路径 + 双输出', ha='center', fontsize=28, fontweight='bold', color='#2e7d32')
ax.text(11.8, 2.15, '联合优化离线/在线性能', ha='center', fontsize=24)

infer_box = FancyBboxPatch((8.3, 0.8), 7.0, 0.8, boxstyle="round,pad=0.1",
                           edgecolor='#4C72B0', facecolor='#e3f2fd', linewidth=3)
ax.add_patch(infer_box)
ax.text(11.8, 1.4, '推理：在线路径 + cache', ha='center', fontsize=28, fontweight='bold', color='#4C72B0')
ax.text(11.8, 0.95, '低延迟流式输出', ha='center', fontsize=24)


plt.tight_layout()
plt.savefig('docs/dhoom_architecture.png', dpi=150, bbox_inches='tight')
plt.close()
print("DHOOM architecture diagram saved to docs/dhoom_architecture.png")

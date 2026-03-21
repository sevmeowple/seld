# ResNet-Conformer-DHOOM 实现报告

**模型名称**: ResNet-Conformer-DHOOM
**实现状态**: Phase 1 & Phase 2 已完成
**日期**: 2026-03-03

---

## 实现概览

ResNet-Conformer-DHOOM 是一个双路径混合离线-在线模型，实现了逐层特征交互：

- **ResNet 编码器**: 独立的 offline/online 路径，BasicBlock 级交互
- **Conformer 编码器**: 独立的 offline/online 层，层间交替融合
- **统一接口**: 支持联合训练 (`forward(x)`) 和流式推理 (`forward(x, cache)`)

---

## 架构详解

### Phase 2: ResNet BasicBlock 级交互 ✅

**核心设计** (参考 `ResnetConformer_sed_doa_nopool_dual_online_offline_TS`):

```python
# 每个 Layer 内的交互模式 (以 Layer 2 为例)
offline_prev = offline_feat   # 24 channels
online_prev = online_feat     # 24 channels

# Block 0: 两者都使用上一轮的特征（同维度）
offline_feat = resnet.layer2[0](offline_prev + online_prev)      # -> 48 channels
online_feat = online_resnet.layer2[0](online_prev + offline_prev)  # -> 48 channels

# Block 1: 同样模式
offline_prev = offline_feat   # 48 channels
online_prev = online_feat     # 48 channels
offline_feat = resnet.layer2[1](offline_prev + online_prev)      # -> 48 channels
online_feat = online_resnet.layer2[1](online_prev + offline_prev)  # -> 48 channels
```

**关键洞察**: 在每个 BasicBlock 之前，使用**上一轮的两个特征**进行融合，而不是使用刚更新的特征。这保证了融合时两个特征维度始终匹配。

**完整的 ResNet 交互流程**:
```
Input: (B, 7, T, 64)
├── Conv1: offline + online 各自处理 (B, 24, T, 64)
├── Layer 1 (2 blocks, 24 channels):
│   ├── Block 0: offline = f(offline + online), online = f(online + offline)
│   └── Block 1: offline = f(offline + online), online = f(online + offline)
│   └── MaxPool: (B, 24, T, 16)
├── Layer 2 (2 blocks, 48 channels):
│   ├── Block 0: offline = f(offline + online), online = f(online + offline)
│   └── Block 1: offline = f(offline + online), online = f(online + offline)
│   └── MaxPool: (B, 48, T, 4)
├── Layer 3 (2 blocks, 96 channels):
│   ├── Block 0: offline = f(offline + online), online = f(online + offline)
│   └── Block 1: offline = f(offline + online), online = f(online + offline)
│   └── MaxPool: (B, 96, T, 2)
├── Layer 4 (2 blocks, 192 channels):
│   ├── Block 0: offline = f(offline + online), online = f(online + offline)
│   └── Block 1: offline = f(offline + online), online = f(online + offline)
└── Conv5: (B, 256, T, 2)
```

### Phase 1: Conformer 层间交互 ✅

**核心设计**:
```python
for i in range(num_conformer_layers):  # 8 layers
    # Offline 路径: 使用上一轮融合特征
    offline_feat = offline_layers[i](offline_feat + online_feat, mask=None)

    # Online 路径: 使用 Offline 输出更新
    online_feat = online_layers[i](online_feat + offline_feat, mask=causal_mask)
```

**注意**: Conformer 使用与 ResNet **相反**的交互顺序：
- ResNet: 两个路径同时使用上一轮特征
- Conformer: Offline 先更新，Online 使用更新后的 Offline

### 双模式接口

**1. 联合训练模式** (`cache=None`):
```python
offline_pred, online_pred = model(x)
# Returns: ((B, T//5, 52), (B, T//5, 52))
```

**2. 流式推理模式** (`cache` provided):
```python
pred, (next_resnet_cache, next_conformer_cache) = model(
    x_chunk, resnet_cache, conformer_cache
)
# Returns: ((B, T//5, 52), (caches,))
```

---

## 关键实现细节

### 1. 维度匹配策略

**挑战**: Layer2/Layer3/Layer4 的第一个 BasicBlock 会改变通道数（downsample）

**解决方案**:
```python
# 保存上一轮特征
offline_prev = offline_feat  # e.g., 24 channels
online_prev = online_feat    # e.g., 24 channels

# 两个路径都使用上一轮特征进行融合
offline_feat = layer[0](offline_prev + online_prev)    # -> 48 channels
online_feat = online_layer[0](online_prev + offline_prev)  # -> 48 channels
```

### 2. 归一化层选择

- **Online ResNet**: `LayerNorm` (用于因果/流式场景)
- **Offline ResNet**: `LayerNorm` (当前代码库实现)
- **参考代码差异**: 参考代码中 Offline 使用 `BatchNorm`，Online 使用 `LayerNorm`

### 3. 独立路径实现

```python
# 两个独立的 ResNet 实例
self.resnet = resnet18_nopool(in_channel=in_channel)        # Offline
self.online_resnet = resnet18_nopool(in_channel=in_channel)  # Online

# 独立的 Conformer layers
self.offline_conformer_layers = nn.ModuleList([...])  # 8 layers
self.online_conformer_layers = nn.ModuleList([...])   # 8 layers
```

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `src/seld_v2/models/resnet_conformer_dhoom.py` | 主模型实现 (350+ 行) |
| `src/seld_v2/models/__init__.py` | 模型导出 |
| `src/config/schema.py` | 配置模型注册 |
| `src/seld_v2/train_v2.py` | 训练脚本适配 |
| `configs/train_resnet_conformer_dhoom.toml` | 训练配置 |
| `configs/dynamic_test/resnet_conformer_dhoom_chunk_4.toml` | 流式测试配置 |

---

## 验证结果

```bash
# 联合训练前向
Offline pred: torch.Size([2, 20, 52])  ✓
Online pred:  torch.Size([2, 20, 52])  ✓
Total parameters: 23,429,688

# 流式推理
Streaming pred: torch.Size([1, 2, 52])  ✓
Next ResNet cache: tuple ✓
Next Conformer cache: 8 layers ✓
```

---

## 使用方式

### 训练
```bash
uv run src/seld_v2/train_v2.py --config configs/train_resnet_conformer_dhoom.toml
```

### 流式测试 (固定 chunk [100, 4])
```bash
uv run src/seld_v2/test/test_cache_model_streaming.py \
    --config configs/dynamic_test/resnet_conformer_dhoom_chunk_4.toml
```

### Dynamic Chunk 训练
修改配置文件中的 `use_dynamic_chunk = true` 和 `chunk_candidates`。

---

## 架构对比

| 特性 | 原 DHOOM | ResNet-Conformer-DHOOM |
|------|---------|----------------------|
| **ResNet** | 共享 + 模式切换 | ✅ 独立 online/offline 路径 |
| **ResNet 交互** | ❌ 无 | ✅ BasicBlock 级（8 blocks） |
| **Conformer** | 共享 layers | ✅ 独立 layers |
| **Conformer 交互** | 单层内融合 | ✅ 层间交替融合（8 layers） |
| **流式接口** | ❌ 不支持 | ✅ `forward(x, cache)` |
| **参数量** | ~14M | ~23M |

---

## 参考信息

**参考代码**: `cache_resnet_conformer_dual.ResnetConformer_sed_doa_nopool_dual_online_offline_TS`

**关键区别**:
1. 本实现使用 `LayerNorm` (而非 `BatchNorm`) 保持与现有代码库一致
2. 输入投影层共享 (而非独立)
3. 输出头遵循 DHOOM 风格 (而非单一输出)

---

*报告完成 - Phase 1 & 2 全部实现*

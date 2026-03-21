# DHOOM 逐层交互架构：代码复用与可行性分析报告

**日期**: 2026-03-03
**分析对象**: 将 DHOOM 改造为逐层交互架构（参考 ResnetConformer_sed_doa_nopool_dual_online_offline_TS）

---

## 1. 技术需求总结

### 1.1 导师要求

基于现有 DHOOM 统一架构，实现与参考代码一致的**逐层交互机制**：

1. **ResNet 逐层交互**: 每层 BasicBlock 后交替融合 offline/online 特征
2. **Conformer 逐层交互**: 每层 ConformerBlock 后交替融合
3. **统一接口**: `forward(x)` 联合训练，`forward(x, cache)` 流式推理
4. **实验配置**:
   - 固定 chunk: `att_context_size = [100, 4]`
   - Dynamic chunk: U2-style 训练

### 1.2 当前 DHOOM vs 目标架构对比

| 组件 | 当前 DHOOM | 目标架构（参考代码） | 差异级别 |
|------|-----------|---------------------|---------|
| **ResNet** | 共享 resnet，模式切换 (`noncausal_mode`) | 独立 online/offline 路径 | 🔴 高 |
| **ResNet 融合** | 无交互，分别跑完再融合 | 每层 BasicBlock 交替融合 | 🔴 高 |
| **Conformer** | 共享 layers，mask 区分因果 | 独立 layers 实例 | 🟡 中 |
| **Conformer 融合** | 每层内 left+right 后分别处理 | 每层间交替: offline→online→offline→online | 🟡 中 |
| **流式推理** | `forward()` 同时输出两种结果 | `forward(x, cache)` 只走 online 路径 | 🔴 高 |

---

## 2. 代码可复用性分析

### 2.1 可完全复用的组件 ✅

#### ConformerBlock (无需修改)
**文件**: `src/seld_v2/models/components/conformer.py`

```python
class ConformerBlock(nn.Module):
    # 已有功能: cache 支持、mask 支持
    def forward(self, x, mask=None, cache=None):
        ...
```

**复用理由**:
- 已支持 cache-based streaming（返回 `next_cache`）
- 已支持 mask-based attention 控制
- 结构完整（FF → Attention → Conv → FF）
- 参考代码使用相同结构

**复用方式**: 直接实例化两组:
```python
self.online_conformer_layers = nn.ModuleList([...])  # 8 layers
self.offline_conformer_layers = nn.ModuleList([...])  # 8 layers
```

#### CausalAttention with Cache
**复用理由**:
- WeNet-style Q-from-input/KV-from-context 已正确实现
- Relative position encoding 支持 Q/K 长度不匹配
- Cache 切片逻辑 `[100, 49]` 格式兼容

### 2.2 需要修改的组件 🟡

#### ResNet (重大修改)
**当前设计问题**:
- 使用 `noncausal_mode()` context manager 切换模式
- 单层 ResNet 跑两次，无法中间交互

**目标设计**:
```python
# 参考代码模式
self.online = ResnetConformer_sed_doa_nopool(...)  # 完整 online 路径
self.resnet = resnet18_nopool(...)  # offline ResNet (独立)
```

**修改方案**:
方案 A: 创建两个独立的 ResNet 实例
- 优点: 简单直接，无状态干扰
- 缺点: 参数量 2x

方案 B: 复用 ResNet 类，但独立实例
- 保持当前 `resnet18_nopool()` 工厂函数
- DHOOM 内创建两个实例: `self.resnet_offline`, `self.resnet_online`

**推荐**: 方案 B，当前 ResNet 已实现 cache 支持，可复用

### 2.3 需要新增的组件 🔴

#### 1. ResNet 逐层交互逻辑
参考代码片段:
```python
# Layer 1, Block 0
offline_fea1 = self.resnet.layer1[0](offline_fea + online_fea)
online_fea1 = self.online.resnet.layer1[0](online_fea + offline_fea1)

# Layer 1, Block 1
offline_fea1 = self.resnet.layer1[1](offline_fea1 + online_fea1)
online_fea1 = self.online.resnet.layer1[1](online_fea1 + offline_fea1)
```

**复杂度**: 高
- 需要手动展开 ResNet 的前向传播
- 不再使用 `self.resnet(x)` 简单调用
- 每层 layer 有 2 个 BasicBlock，共 4 layers = 8 次交互

#### 2. Conformer 逐层交互逻辑
参考代码片段:
```python
for i in range(8):
    offline_out = self.conformer_layers[i](offline_out + online_out)
    online_out = self.online.conformer_layers[i](online_out + offline_out)
```

**复杂度**: 中
- 循环内交替更新
- 每层后残差融合 (`+`)

#### 3. 统一接口实现
参考代码模式:
```python
def forward(self, x, resnet_cache=None, conformer_cache=None):
    if resnet_cache is None and conformer_cache is None:
        # 联合训练模式
        return [pred]  # 或 [pred, hidden_pairs] for distill
    else:
        # 纯流式推理
        return pred, (next_resnet_cache, next_conformer_cache)
```

**关键差异**:
- 当前 DHOOM 的 `forward()` 返回 `(offline_pred, streaming_pred)` 元组
- 参考代码返回列表，支持蒸馏损失

---

## 3. 代码复杂度评估

### 3.1 修改范围量化

| 文件 | 修改行数估计 | 修改类型 |
|------|-------------|---------|
| `resnet_dhoom.py` | ~150 行重写 | 核心架构重写 |
| `experiment.py` | ~20 行 | 适配新的输出格式 |
| `train_v2.py` | ~30 行 | 损失计算适配 |
| 新增配置文件 | 2 个文件 | chunk_100_4, dynamic_chunk |

### 3.2 关键复杂度点

#### 🔴 高复杂度: ResNet 逐层展开

当前 DHOOM:
```python
left_2d = self.resnet(x)  # 1 行
with self.resnet.noncausal_mode():
    right_2d = self.resnet(x)  # 1 行
```

目标代码（估计 ~60 行）:
```python
# Conv1 初始化
online_fea = self.online.resnet.conv1(x)
online_fea = self._apply_ln_and_relu(online_fea, self.online.resnet)
offline_fea = self.resnet.conv1(x)
offline_fea = self._apply_bn_and_relu(offline_fea, self.resnet)

# Layer 1 (2 blocks)
offline_fea = self.resnet.layer1[0](offline_fea + online_fea)
online_fea = self.online.resnet.layer1[0](online_fea + offline_fea)
offline_fea = self.resnet.layer1[1](offline_fea + online_fea)
online_fea = self.online.resnet.layer1[1](online_fea + offline_fea)
offline_fea = self.resnet.maxpool1(offline_fea)
online_fea = self.online.resnet.maxpool1(online_fea)

# Layer 2, 3, 4 类似...
```

**风险点**:
- 容易出错：忘记 apply norm/relu
- 维度不匹配：maxpool 后特征尺寸变化
- 权重初始化：两个 ResNet 实例需要独立初始化

#### 🟡 中复杂度: 训练流程适配

参考代码支持知识蒸馏（`use_hidden_distill`, `use_apc_distill`），导师要求设为 False，但仍需适配输出格式：

```python
# 当前 DHOOM
return offline_pred, streaming_pred

# 参考代码
results = [pred]
if self.use_hidden_distill:
    results.append((offline_conformer_outputs, online_conformer_outputs))
return results
```

需要修改训练循环处理列表返回值。

#### 🟢 低复杂度: 配置文件

仅需复制现有配置并修改 `att_context_size`。

---

## 4. 可行性方案对比

### 方案 A: 完全复刻参考架构（推荐度: ⭐⭐⭐）

**实现方式**:
1. 创建独立的 `self.online` 属性（完整 ResnetConformer）
2. 完全重写 ResNet 前向，逐层展开交互
3. 完全重写 Conformer 前向，逐层交替
4. 实现 `forward(x)` 和 `forward(x, cache)` 双模式

**优点**:
- 与参考代码一致，便于对比实验
- 架构清晰，交互逻辑明确
- 支持 future extension（如添加蒸馏损失）

**缺点**:
- 工作量大（~3-4 天）
- 需要仔细验证每层的维度匹配
- 需要重新训练（无法复用当前 checkpoint）

**工作量**: 高（3-4 天）

---

### 方案 B: 渐进式改造（推荐度: ⭐⭐⭐⭐）

**实现方式**:
阶段 1: 先实现 Conformer 逐层交互（保持 ResNet 共享）
阶段 2: 再添加 ResNet 逐层交互（如果需要）

**阶段 1 代码变化**:
```python
# 当前：共享 layers
self.conformer_layers = nn.ModuleList([...])

# 目标：独立 layers，每层后融合
self.online_conformer_layers = nn.ModuleList([...])
self.offline_conformer_layers = nn.ModuleList([...])

# 前向逻辑变化
for i in range(num_layers):
    right = self.offline_layers[i](left + right, mask=None)
    left = self.online_layers[i](left + right, mask=causal_mask)
```

**优点**:
- 分阶段验证，降低风险
- 阶段 1 工作量小（1 天）
- 可对比 ResNet 交互的必要性

**缺点**:
- 可能需要两轮实验验证

**工作量**: 中（1-2 天）

---

### 方案 C: 最小改动方案（推荐度: ⭐⭐）

**实现方式**:
- 仅修改 Conformer 部分：从 "每层内融合" 改为 "层间融合"
- ResNet 保持现状（共享 + 模式切换）

**当前 DHOOM Conformer**:
```python
for i, layer in enumerate(self.conformer_layers):
    merged = left + right
    right = layer(merged, mask=None)      # 用 merged
    left = layer(fuse(merged, right), mask=causal_mask)  # 用 merged + right
```

**目标 Conformer**:
```python
for i in range(num_layers):
    right = self.offline_layers[i](right + left, mask=None)
    left = self.online_layers[i](left + right, mask=causal_mask)
```

**区别**:
- 当前: 共享 layer 实例，内部两次调用
- 目标: 独立 layer 实例，层间传递

**优点**:
- 工作量最小（半天）
- 核心逻辑改变（逐层交互）

**缺点**:
- ResNet 部分未按导师要求实现
- 可能无法达到最佳效果

**工作量**: 低（半天-1 天）

---

## 5. 推荐方案与实施计划

### 推荐: 方案 B（渐进式改造）

理由:
1. **风险可控**: 分阶段验证，每步都有 checkpoint 可回退
2. **与导师要求一致**: 最终实现与参考代码等效
3. **工作量合理**: 阶段 1 快速出成果，阶段 2 视情况决定

### 实施步骤

#### Phase 1: Conformer 逐层交互（Day 1）

**Step 1**: 修改 `resnet_dhoom.py` 初始化
```python
# 添加独立的 online 路径
self.online_resnet = resnet18_nopool(in_channel=in_channel)
self.online_projection = nn.Sequential(...)

# 改为独立 conformer layers
self.online_conformer_layers = nn.ModuleList([...])
self.offline_conformer_layers = nn.ModuleList([...])
```

**Step 2**: 重写 Conformer 前向
```python
# 保持 ResNet 部分不变
left_2d = self.resnet(x)
with self.resnet.noncausal_mode():
    right_2d = self.resnet(x)

# 新的 Conformer 逐层交互
left = self.input_projection(flatten(left_2d))
right = self.input_projection(flatten(right_2d))

for i in range(num_layers):
    right = self.offline_conformer_layers[i](right + left, mask=None)
    left = self.online_conformer_layers[i](left + right, mask=causal_mask)
```

**Step 3**: 测试 & 验证
- 维度检查: `(B, T, 256)` 保持不变
- 梯度检查: 确保两个路径都有梯度
- 快速训练: 1-2 个 epoch 验证收敛

#### Phase 2: ResNet 逐层交互（Day 2-3，可选）

如果 Phase 1 效果不佳，或导师明确要求，则进行 ResNet 改造:

**Step 1**: 展开 ResNet 前向
- 手动实现 conv1 → layer1[0] → layer1[1] → maxpool1 → ...
- 每层后添加 `offline + online` 融合

**Step 2**: 验证等价性
- 确保改造前后输出维度一致
- 检查参数量变化

#### Phase 3: 统一接口 & 流式推理（Day 3-4）

**Step 1**: 实现 `forward(x, resnet_cache, conformer_cache)`
```python
if resnet_cache is not None:
    # 纯流式：只用 online 路径
    conv_out, next_resnet_cache = self.online_resnet(x, resnet_cache)
    ...
    return pred, (next_resnet_cache, next_conformer_cache)
```

**Step 2**: 配置文件
- `configs/dynamic_test/dhoom_chunk_100_4.toml`
- `configs/train_dhoom_layerwise.toml`

**Step 3**: 完整实验
- 固定 chunk [100, 4] 测试
- Dynamic chunk 训练

---

## 6. 风险评估与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|-------|------|---------|
| **维度不匹配** | 高 | 高 | 每步打印 shape，参考代码对照 |
| **梯度消失/爆炸** | 中 | 高 | 使用 gradient clipping，监控梯度norm |
| **内存不足** | 中 | 中 | 两个 ResNet 同时前向，显存 x2，可用 gradient checkpointing |
| **训练不收敛** | 中 | 高 | 先小数据验证，对比 baseline |
| **无法复用 checkpoint** | 高 | 中 | 预期内，需重新训练 |
| **与参考代码行为不一致** | 中 | 高 | 单元测试对比中间输出 |

---

## 7. 代码修改详细清单

### 必须修改的文件

```
src/seld_v2/models/resnet_dhoom.py
├── __init__()
│   ├── 添加 self.online_resnet
│   ├── 添加 self.online_projection
│   ├── 修改 conformer_layers → offline_conformer_layers
│   ├── 添加 online_conformer_layers
│   └── 添加 use_hidden_distill 等标志（设为 False）
│
├── forward()
│   ├── 分支 1: cache is None（联合训练）
│   │   ├── ResNet 部分（可选展开）
│   │   └── Conformer 逐层交互循环
│   │
│   └── 分支 2: cache is not None（流式推理）
│       ├── online_resnet(x, resnet_cache)
│       └── online_conformer_layers(x, conformer_cache)
│
└── 新增 get_initial_cache_*() 方法
```

### 需要适配的文件

```
src/seld_v2/training/experiment.py
└── 可能需适配新的返回值格式

src/seld_v2/train_v2.py
└── 损失计算适配（如果返回列表而非元组）

configs/
├── train_dhoom_layerwise.toml
└── dynamic_test/dhoom_chunk_100_4.toml
```

---

## 8. 总结

### 技术可行性: ✅ 可行

基于现有代码库，实现导师要求的逐层交互架构**技术上完全可行**。

### 关键成功因素

1. **ConformerBlock 可完全复用** - 已支持 cache 和 mask
2. **ResNet 可实例化两份** - 独立 online/offline 路径
3. **参考代码清晰** - 交互逻辑明确，可直接参考

### 推荐行动计划

1. **立即开始 Phase 1**（今天）
   - 实现 Conformer 逐层交互
   - 保持 ResNet 不变

2. **验证后决定 Phase 2**（明天）
   - 如果效果提升明显，继续 ResNet 改造
   - 如果效果一般，与导师讨论是否需要 ResNet 交互

3. **添加流式推理支持**（后天）
   - 实现 cache 接口
   - 跑通 chunk [100, 4] 实验

### 预期结果

- **模型效果**: 流式路径性能提升（参考代码显示优于独立训练）
- **训练速度**: 略有下降（两个路径前向传播）
- **推理速度**: 不变（流式模式只用 online 路径）
- **代码复杂度**: 增加，但结构清晰

---

*报告完成*

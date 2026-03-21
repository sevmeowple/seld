# seld_v2 WeNet 对齐修改报告 — P0 + P1

**日期**: 2026-02-23
**提交**: `373cf1d` (align conformer with WeNet: P0+P1 fixes)
**目标**: 对齐 WeNet conformer 实现，缩小与 0.40 SELD 基线的差距

---

## 修改总览

| 优先级 | 修改项 | 文件 | 影响 |
|--------|--------|------|------|
| P0 | Q/K/V 投影添加 bias | `conformer.py:24-25` | 提升投影表达能力 |
| P0 | Encoder 末尾添加 LayerNorm | `resnet_conformer.py:60,128,151` | 稳定 encoder 输出分布 |
| P0 | 梯度裁剪 clip=5.0 | `train_epoch.py:44` | 防止梯度爆炸 |
| P1 | Conv module 改用 BatchNorm1d | `conformer.py:111,126` | 对齐论文标准，跨 batch 正则化 |
| P1 | Attention dropout 移至 weights | `conformer.py:75` | 结构化正则，防止 attention 过拟合 |
| P1 | Post-softmax NaN 保护 | `conformer.py:73-74` | 全 mask 行安全处理 |

---

## P0-1: Q/K/V 投影添加 bias

**文件**: `src/seld_v2/models/components/conformer.py:24-25`
**原因**: WeNet 的 Q/K/V 投影均使用 `bias=True`，seld_v2 原实现为 `bias=False`，缺少偏置会降低线性投影的表达能力。

```diff
 # CausalAttention.__init__
-        self.to_q = nn.Linear(dim, inner_dim, bias=False)
-        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
+        self.to_q = nn.Linear(dim, inner_dim, bias=True)
+        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=True)
```

**影响**: 每层增加 `inner_dim * 3` 个参数（Q bias + K bias + V bias）。8 层 × 256 dim × 3 = 6144 参数，占比极小。

---

## P0-2: Encoder 末尾添加 LayerNorm

**文件**: `src/seld_v2/models/resnet_conformer.py:60,128,151`
**原因**: WeNet encoder 在所有 conformer 层之后、输出之前有 `after_norm = LayerNorm`。缺少这一层会导致 encoder 输出分布不稳定，下游 SED/DOA 线性头更难优化。

```diff
 # ResnetConformer.__init__
         ])
+        self.after_norm = nn.LayerNorm(encoder_dim)
         self.t_pooling = nn.MaxPool1d(kernel_size=5)
```

```diff
 # _forward_offline
         for layer in self.conformer_layers:
             conformer_out = layer(conformer_out, mask=chunk_mask)
+        conformer_out = self.after_norm(conformer_out)

         pred = self._predict(conformer_out)
```

```diff
 # _forward_streaming
             next_layer_caches.append(next_cache)
+        conformer_out = self.after_norm(conformer_out)

         pred = self._predict(conformer_out)
```

**影响**: offline 和 streaming 两条路径均已添加，确保推理一致性。

---

## P0-3: 梯度裁剪

**文件**: `src/seld_v2/training/train_epoch.py:44`
**原因**: Conformer 的深层残差结构容易出现梯度爆炸，WeNet 使用 `clip_grad_norm_(5.0)` 作为标准配置。seld_v2 原实现完全没有梯度裁剪。

```diff
 # train_one_epoch
         loss.backward()
+        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
         optimizer.step()
```

**影响**: 训练稳定性显著提升，尤其在学习率较高的 warmup 阶段。

---

## P1-1: Conv Module 改用 BatchNorm1d（post-depthwise-conv 位置）

**文件**: `src/seld_v2/models/components/conformer.py:100-133`
**原因**: 两处差异——

1. **Norm 类型**: 原实现用 `LayerNorm`，Conformer 论文和 WeNet 均用 `BatchNorm1d`。BatchNorm 提供跨 batch 的归一化，有额外正则化效果。
2. **Norm 位置**: 原实现将 LayerNorm 放在 conv module 最前面（pre-norm），WeNet 将 BatchNorm 放在 depthwise conv 之后（post-conv）。post-conv 位置归一化了卷积输出，更有利于后续 activation 和 pointwise conv。

```diff
 class ConformerConvModule(nn.Module):
     def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
         super().__init__()
         inner_dim = dim * expansion_factor
         padding = (kernel_size - 1, 0)
-        self.norm = nn.LayerNorm(dim)
         self.conv1 = nn.Conv1d(dim, inner_dim * 2, 1)
         self.glu = GLU(dim=1)
         self.depth_conv = DepthWiseConv1dWithCache(
             inner_dim, inner_dim, kernel_size=kernel_size, padding=padding,
         )
+        self.bn = nn.BatchNorm1d(inner_dim)
         self.swish = Swish()
         self.conv2 = nn.Conv1d(inner_dim, dim, 1)
         self.dropout = nn.Dropout(dropout)

     def forward(self, x, cache=None):
-        x = self.norm(x)
         x = rearrange(x, 'b n c -> b c n')
         x = self.conv1(x)
         x = self.glu(x)
         if cache is None:
             x = self.depth_conv(x)
         else:
             x, next_cache = self.depth_conv(x, cache)
+        x = self.bn(x)
         x = self.swish(x)
         x = self.conv2(x)
         ...
```

**操作顺序变化**:
```
旧: LayerNorm → Transpose → Conv1 → GLU → DepthConv → Swish → Conv2 → Transpose → Dropout
新: Transpose → Conv1 → GLU → DepthConv → BatchNorm → Swish → Conv2 → Transpose → Dropout
```

**影响**: state_dict key 从 `conv.norm.{weight,bias}` 变为 `conv.bn.{weight,bias,running_mean,running_var,num_batches_tracked}`，旧 checkpoint 不兼容，需重新训练。

---

## P1-2: Attention Dropout 移至 Weights + Post-Softmax NaN 保护

**文件**: `src/seld_v2/models/components/conformer.py:72-78`
**原因**:

1. **Dropout 位置**: 原实现在输出投影之后 dropout，WeNet 在 softmax 之后（attention weights 上）dropout。后者是结构化正则化，迫使模型不过度依赖特定的 attention pattern。
2. **NaN 保护**: 当 mask 导致某行全为 `-inf` 时，softmax 输出 NaN。WeNet 在 softmax 后再次 `masked_fill(0.0)` 将其归零。

```diff
         attn = dots.softmax(dim=-1)
+        if mask is not None:
+            attn = attn.masked_fill(~mask, 0.0)
+        attn = self.dropout(attn)
         out = einsum('b h i j, b h j d -> b h i d', attn, v)
         out = rearrange(out, 'b h n d -> b n (h d)')
         out = self.to_out(out)
-        out = self.dropout(out)
```

**影响**: dropout 作用对象从 `(B, T, D)` 的输出向量变为 `(B, H, T_q, T_k)` 的注意力矩阵，正则化粒度更细。

---

## Checkpoint 兼容性

本次修改导致以下 state_dict key 变化，**旧 checkpoint 无法直接加载**：

| 变化类型 | Key |
|----------|-----|
| 新增 | `after_norm.weight`, `after_norm.bias` |
| 新增 | `conformer_layers.*.attn.fn.to_q.bias` |
| 新增 | `conformer_layers.*.attn.fn.to_kv.bias` |
| 新增 | `conformer_layers.*.conv.bn.{weight,bias,running_mean,running_var}` |
| 删除 | `conformer_layers.*.conv.norm.{weight,bias}` |

需从头训练新模型。

---

## 验证

模型 forward pass 测试通过（offline + streaming）：

```
Offline output shape:  torch.Size([2, 10, 52])
Streaming output shape: torch.Size([2, 3, 52])
```

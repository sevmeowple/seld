# SELD v2 - 重构版本

## 目录结构

```
seld_v2/
├── data/           # 数据处理
│   ├── transforms.py    # 数据预处理函数
│   └── dataset.py       # 数据集类
├── losses/         # 损失函数
│   ├── task_loss.py         # SED+DOA任务损失
│   ├── distillation_loss.py # 知识蒸馏损失
│   └── attention_loss.py    # 注意力损失
├── metrics/        # 评估指标
│   └── result_collector.py  # 结果收集
├── models/         # 模型定义
│   └── (待迁移)
└── training/       # 训练流程
    ├── trainer.py      # 训练器
    ├── evaluator.py    # 评估器
    └── checkpoint.py   # 检查点管理
```

## 重构进度

- [x] Task 0.1: 创建目录结构
- [x] Task 0.2: 基础 __init__.py
- [ ] Task 1.1: data/transforms.py
- [ ] Task 1.2: data/dataset.py
- [ ] Task 2.1: losses/task_loss.py
- [ ] ...

## 使用方式

重构完成后,使用新的训练脚本:
```bash
python train_v2.py --config configs/exp1.toml
```

原有代码保持不变,在 `src/seld/` 目录。

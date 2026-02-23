from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


class Mixup(nn.Module):
    """Manifold Mixup: 在 ResNet 层级做 mixup 数据增强"""

    def __init__(
        self,
        model: nn.Module,
        mix_probability: float = 1.1,
        alpha: float = 1.0,
        batch_first: bool = True,
    ):
        super().__init__()
        self.model = model
        self.mix_probability = mix_probability
        self.alpha = alpha
        self.batch_first = batch_first
        self.indices: torch.Tensor = torch.empty(0)
        self.lam: float = 0.0

        self.module_list: List[nn.Module] = []
        for n, m in self.model.named_modules():
            if n[:-1] == "layer":
                self.module_list.append(m)

    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[float]]:
        batch_size = x.shape[0]
        labels: List[torch.Tensor] = [label]
        weights: List[float] = []

        if np.random.uniform(0, 1) < self.mix_probability:
            self.indices = torch.randperm(batch_size)
            self.lam = float(np.random.beta(self.alpha, self.alpha))
            weights.append(self.lam)
            labels.append(label[self.indices])

            mixed_layer_idx = -1
            if mixed_layer_idx == -1:
                x = (1 - self.lam) * x + self.lam * x[self.indices]
                y = self.model(x)
            else:
                hook = self.module_list[mixed_layer_idx].register_forward_hook(
                    self._hook_modify
                )
                y = self.model(x)
                hook.remove()
        else:
            y = self.model(x)
            weights.append(1.0)
            labels.append(label)

        return y, labels, weights

    def _hook_modify(
        self, module: nn.Module, input: Tuple, output: torch.Tensor
    ) -> torch.Tensor:
        if not isinstance(output, torch.Tensor):
            raise NotImplementedError("unknown output for module")
        if self.batch_first:
            return (1 - self.lam) * output + self.lam * output[self.indices]
        return (1 - self.lam) * output + self.lam * output[:, self.indices]

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn


class DualHeadSedDoaLoss(nn.Module):
    """Wraps a base SedDoaLoss to handle dual-head (offline, streaming) output.

    If output is a tuple: loss = base(offline, target) + weight * base(streaming, target)
    If output is a single tensor: falls back to base(output, target).
    """

    def __init__(self, base_loss: nn.Module, streaming_weight: float = 1.0):
        super().__init__()
        self.base_loss = base_loss
        self.streaming_weight = streaming_weight

    def forward(
        self, output: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], target: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(output, tuple):
            offline_out, streaming_out = output
            return self.base_loss(offline_out, target) + self.streaming_weight * self.base_loss(streaming_out, target)
        return self.base_loss(output, target)

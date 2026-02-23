from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def exists(val: object) -> bool:
    return val is not None


def default(val: object, d: object) -> object:
    return val if exists(val) else d


def calc_same_padding(kernel_size: int) -> Tuple[int, int]:
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in: int, chan_out: int, kernel_size: int, padding: Tuple[int, int]):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)

class DepthWiseConv1dWithCache(nn.Module):
    def __init__(self, chan_in: int, chan_out: int, kernel_size: int, padding: Tuple[int, int]):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups=chan_in)
        self.cache_drop_size: Optional[int] = None
        self.padding = padding

    def update_cache(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if cache is None:
            x = F.pad(x, self.padding)
            return x, None
        x = torch.cat([cache, x], dim=-1)
        if self.cache_drop_size:
            next_cache = x[..., :-self.cache_drop_size]
        else:
            next_cache = x[..., -self.kernel_size + 1:]
        return x, next_cache

    def forward(self, x: torch.Tensor, cache: Optional[torch.Tensor] = None) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x, next_cache = self.update_cache(x, cache)
        x = self.conv(x)
        if cache is None:
            return x
        return x, next_cache

class Scale(nn.Module):
    def __init__(self, scale: float, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.norm(x)
        return self.fn(x, **kwargs)

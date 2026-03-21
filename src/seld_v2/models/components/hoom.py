from __future__ import annotations

import torch
import torch.nn as nn


class CausalConv2d(nn.Module):
    """Conv2d with causal (left) padding on the time axis."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=(kernel_size // 2, kernel_size // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F) — pad time axis causally
        x = nn.functional.pad(x, (0, 0, self.pad, 0))  # left-pad time
        x = self.conv(x)
        x = x[:, :, :x.shape[2] - self.pad] if self.pad > 0 else x
        return x


def _conv_unit(in_ch: int, out_ch: int, kernel_size: int, dropout: float, causal: bool):
    """Conv2d → GroupNorm (equivalent to LayerNorm) → ReLU → Dropout. Causal or regular."""
    if causal:
        return nn.Sequential(
            CausalConv2d(in_ch, out_ch, kernel_size),
            nn.GroupNorm(num_groups=1, num_channels=out_ch), nn.ReLU(), nn.Dropout2d(dropout),
        )
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
        nn.GroupNorm(num_groups=1, num_channels=out_ch), nn.ReLU(), nn.Dropout2d(dropout),
    )


class CCANBlock(nn.Module):
    """Causal Context-Aware Network block (2D dual-path conv).

    Block 1 (is_first=True):  left input = concat(x, right_out) on channel dim
    Block n>=2 (is_first=False): left input = x + right_out (element-wise add)
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1, is_first: bool = False):
        super().__init__()
        self.is_first = is_first
        self.right_conv = _conv_unit(in_ch, out_ch, kernel_size, dropout, causal=False)
        if is_first:
            # Block 1: left takes concat(in_ch, out_ch)
            self.left_conv = _conv_unit(in_ch + out_ch, out_ch, kernel_size, dropout, causal=True)
        else:
            # Block n>=2: left takes x_proj + right_out, both out_ch
            self.left_conv = _conv_unit(out_ch, out_ch, kernel_size, dropout, causal=True)
            if in_ch != out_ch:
                self.input_proj = nn.Conv2d(in_ch, out_ch, 1)
            else:
                self.input_proj = None

    def forward(self, left: torch.Tensor | None, right: torch.Tensor | None, x: torch.Tensor):
        if left is not None:
            x = left + right
        right_out = self.right_conv(x)
        if self.is_first:
            left_out = self.left_conv(torch.cat([x, right_out], dim=1))
        else:
            x_proj = self.input_proj(x) if self.input_proj is not None else x
            left_out = self.left_conv(x_proj + right_out)
        return left_out, right_out


class BUANBlock(nn.Module):
    """Bidirectional-Unidirectional Aggregation Network block (dual-path GRU).

    Block 1 (is_first=True):  left input = concat(x, right_out) → proj → GRU
    Block m>=2 (is_first=False): left input = x + right_out (element-wise add) → GRU
    """

    def __init__(self, dim: int, dropout: float = 0.1, is_first: bool = False):
        super().__init__()
        self.is_first = is_first
        self.right_gru = nn.GRU(dim, dim // 2, bidirectional=True, batch_first=True)
        self.right_norm = nn.LayerNorm(dim)
        self.right_drop = nn.Dropout(dropout)
        if is_first:
            # Block 1: concat(dim, dim) = 2*dim → project to dim
            self.left_proj = nn.Linear(2 * dim, dim)
        self.left_gru = nn.GRU(dim, dim, bidirectional=False, batch_first=True)
        self.left_norm = nn.LayerNorm(dim)
        self.left_drop = nn.Dropout(dropout)

    def forward(self, left: torch.Tensor | None, right: torch.Tensor | None, x: torch.Tensor):
        if left is not None:
            x = left + right
        right_out, _ = self.right_gru(x)
        right_out = self.right_drop(self.right_norm(right_out))
        if self.is_first:
            left_in = self.left_proj(torch.cat([x, right_out], dim=-1))
        else:
            left_in = x + right_out
        left_out, _ = self.left_gru(left_in)
        left_out = self.left_drop(self.left_norm(left_out))
        return left_out, right_out

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
    """Conv2d → BN → ReLU → Dropout. Causal or regular."""
    if causal:
        return nn.Sequential(
            CausalConv2d(in_ch, out_ch, kernel_size),
            nn.BatchNorm2d(out_ch), nn.ReLU(), nn.Dropout2d(dropout),
        )
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
        nn.BatchNorm2d(out_ch), nn.ReLU(), nn.Dropout2d(dropout),
    )


class CCANBlock(nn.Module):
    """Causal Context-Aware Network block (2D dual-path conv).

    right_conv: regular Conv2d path
    left_conv: causal Conv2d path, takes cat(input, right) on channel dim
    Output: left + right
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.right_conv = _conv_unit(in_ch, out_ch, kernel_size, dropout, causal=False)
        # left takes cat(in_ch, out_ch) → needs input_proj
        self.left_conv = _conv_unit(in_ch + out_ch, out_ch, kernel_size, dropout, causal=True)

    def forward(self, left: torch.Tensor | None, right: torch.Tensor | None, x: torch.Tensor):
        """
        For block 1: left=None, right=None, x=input
        For block n≥2: left=prev_left, right=prev_right, x is ignored (merged = left+right)
        Returns (left_out, right_out)
        """
        if left is not None:
            x = left + right  # merged
        right_out = self.right_conv(x)
        left_out = self.left_conv(torch.cat([x, right_out], dim=1))
        return left_out, right_out


class BUANBlock(nn.Module):
    """Bidirectional-Unidirectional Aggregation Network block (dual-path GRU).

    right_gru: BiGRU
    left_gru: UniGRU, takes cat(input, right) on feature dim
    Output: left + right
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.right_gru = nn.GRU(dim, dim // 2, bidirectional=True, batch_first=True)
        self.right_norm = nn.LayerNorm(dim)
        self.right_drop = nn.Dropout(dropout)
        # left takes cat(dim, dim) = 2*dim → project to dim
        self.left_proj = nn.Linear(2 * dim, dim)
        self.left_gru = nn.GRU(dim, dim, bidirectional=False, batch_first=True)
        self.left_norm = nn.LayerNorm(dim)
        self.left_drop = nn.Dropout(dropout)

    def forward(self, left: torch.Tensor | None, right: torch.Tensor | None, x: torch.Tensor):
        """
        For block 1: left=None, right=None, x=input
        For block n≥2: left=prev_left, right=prev_right
        Returns (left_out, right_out)
        """
        if left is not None:
            x = left + right  # merged
        right_out, _ = self.right_gru(x)
        right_out = self.right_drop(self.right_norm(right_out))
        cat_input = self.left_proj(torch.cat([x, right_out], dim=-1))
        left_out, _ = self.left_gru(cat_input)
        left_out = self.left_drop(self.left_norm(left_out))
        return left_out, right_out

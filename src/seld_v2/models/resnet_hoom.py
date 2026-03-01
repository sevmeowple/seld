from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from seld_v2.models.components.hoom import CCANBlock, BUANBlock


class HOOM(nn.Module):
    """HOOM model (Zhang et al. 2024) for SELD.

    Architecture: CCAN blocks (2D) → flatten → linear → BUAN blocks (1D) → 3×MHSA → FC heads
    Layout configured via hoom_layout, e.g. ["ccan", "ccan", "buan", "buan"].
    """

    def __init__(
        self,
        in_channel: int = 7,
        in_dim: int = 64,
        out_dim: int = 39,
        encoder_dim: int = 256,
        num_classes: int = 13,
        hoom_layout: List[str] = ["ccan", "ccan", "buan", "buan"],
        ccan_channels: List[int] = [64, 128],
        freq_pool_sizes: List[int] = [2, 2],
        num_mhsa: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hoom_layout = hoom_layout

        # Build CCAN blocks
        ccan_layers = [l for l in hoom_layout if l == "ccan"]
        assert len(ccan_layers) == len(ccan_channels) == len(freq_pool_sizes)

        self.ccan_blocks = nn.ModuleList()
        self.freq_pools = nn.ModuleList()
        prev_ch = in_channel
        freq_dim = in_dim
        for ch, pool_sz in zip(ccan_channels, freq_pool_sizes):
            self.ccan_blocks.append(CCANBlock(prev_ch, ch, dropout=dropout))
            self.freq_pools.append(nn.MaxPool2d((1, pool_sz)))
            prev_ch = ch
            freq_dim = freq_dim // pool_sz

        # Flatten projection: (B, T, C*F) → (B, T, encoder_dim)
        self.flatten_proj = nn.Linear(prev_ch * freq_dim, encoder_dim)

        # Build BUAN blocks
        buan_layers = [l for l in hoom_layout if l == "buan"]
        self.buan_blocks = nn.ModuleList([
            BUANBlock(encoder_dim, dropout=dropout) for _ in buan_layers
        ])

        # 3 × MHSA + LayerNorm
        self.mhsa_layers = nn.ModuleList()
        self.mhsa_norms = nn.ModuleList()
        for _ in range(num_mhsa):
            self.mhsa_layers.append(nn.MultiheadAttention(encoder_dim, num_heads, dropout=dropout, batch_first=True))
            self.mhsa_norms.append(nn.LayerNorm(encoder_dim))

        # Temporal pooling + output heads
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        self.sed_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, num_classes), nn.Sigmoid(),
        )
        self.doa_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, F) e.g. (B, 7, T, 64)

        # CCAN blocks: chain (left, right) state
        left, right = None, None
        for ccan, pool in zip(self.ccan_blocks, self.freq_pools):
            left, right = ccan(left, right, x if left is None else None)
            # Pool frequency after each block; update x not needed since ccan uses left/right
            left = pool(left)
            right = pool(right)

        out = left + right  # (B, C_last, T, F')

        # Flatten: (B, C, T, F) → (B, T, C*F)
        B, C, T, F = out.shape
        out = out.permute(0, 2, 1, 3).reshape(B, T, C * F)
        out = self.flatten_proj(out)

        # BUAN blocks: chain (left, right) state
        left_1d, right_1d = None, None
        for buan in self.buan_blocks:
            left_1d, right_1d = buan(left_1d, right_1d, out if left_1d is None else None)
        out = left_1d + right_1d

        # MHSA layers with residual + LayerNorm
        for mhsa, norm in zip(self.mhsa_layers, self.mhsa_norms):
            residual = out
            attn_out, _ = mhsa(out, out, out)
            out = norm(attn_out + residual)

        # Temporal pooling + heads
        out = self.t_pooling(out.permute(0, 2, 1)).permute(0, 2, 1)
        sed = self.sed_head(out)
        doa = self.doa_head(out)
        return torch.cat((sed, doa), dim=-1)

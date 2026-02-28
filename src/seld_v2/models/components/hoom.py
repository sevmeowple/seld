from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from seld_v2.models.common import PreNorm, Scale
from seld_v2.models.components.conformer import (
    CausalAttention,
    ConformerConvModule,
    FeedForward,
)


class CCANBlock(nn.Module):
    """Causal Context-Aware Network block.

    Same structure as ConformerBlock but always uses causal (lower-triangular)
    mask — no U2 dynamic chunk logic.
    """

    def __init__(
        self, *, dim: int, dim_head: int = 32, heads: int = 8,
        ff_mult: int = 2, conv_expansion_factor: int = 2,
        conv_kernel_size: int = 7, attn_dropout: float = 0.1,
        ff_dropout: float = 0.1, conv_dropout: float = 0.1,
    ):
        super().__init__()
        # att_context_size=[-1, -1]: unlimited context, no chunk masking
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = CausalAttention(
            dim=dim, dim_head=dim_head, heads=heads,
            dropout=attn_dropout, att_context_size=[-1, -1],
        )
        self.conv = ConformerConvModule(
            dim=dim, expansion_factor=conv_expansion_factor,
            kernel_size=conv_kernel_size, dropout=conv_dropout,
        )
        self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Build causal mask: lower-triangular
        T = x.shape[1]
        causal_mask = torch.tril(
            torch.ones(1, 1, T, T, dtype=torch.bool, device=x.device),
        )
        x = self.ff1(x) + x
        x = self.attn(x, mask=causal_mask) + x
        x = self.conv(x) + x
        x = self.ff2(x) + x
        return self.post_norm(x)


class BUANBlock(nn.Module):
    """Bidirectional-Unidirectional Aggregation Network block.

    Bidirectional GRU captures full context, then unidirectional attention
    (causal mask) aggregates with a feed-forward sublayer.
    """

    def __init__(
        self, *, dim: int, dim_head: int = 32, heads: int = 8,
        attn_dropout: float = 0.1, ff_mult: int = 2, ff_dropout: float = 0.1,
    ):
        super().__init__()
        self.gru = nn.GRU(dim, dim // 2, bidirectional=True, batch_first=True)
        self.gru_proj = nn.Linear(dim, dim)
        self.gru_norm = nn.LayerNorm(dim)

        self.attn = CausalAttention(
            dim=dim, dim_head=dim_head, heads=heads,
            dropout=attn_dropout, att_context_size=[-1, -1],
        )
        self.attn = PreNorm(dim, self.attn)
        self.attn_norm = nn.LayerNorm(dim)

        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.ff = PreNorm(dim, self.ff)
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        causal_mask = torch.tril(
            torch.ones(1, 1, T, T, dtype=torch.bool, device=x.device),
        )

        # Bidirectional GRU + residual
        gru_out, _ = self.gru(x)
        gru_out = self.gru_proj(gru_out)
        x = self.gru_norm(gru_out + x)

        # Unidirectional attention + residual
        x = self.attn_norm(self.attn(x, mask=causal_mask) + x)

        # Feed-forward + residual
        x = self.ff_norm(self.ff(x) + x)
        return x


class HOOMLayer(nn.Module):
    """Single HOOM layer: parallel CCAN + BUAN branches with fusion."""

    def __init__(
        self, *, dim: int, dim_head: int = 32, heads: int = 8,
        ff_mult: int = 2, conv_expansion_factor: int = 2,
        conv_kernel_size: int = 7, attn_dropout: float = 0.1,
        ff_dropout: float = 0.1, conv_dropout: float = 0.1,
        fusion: Literal["concat", "add"] = "concat",
    ):
        super().__init__()
        self.ccan = CCANBlock(
            dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size, attn_dropout=attn_dropout,
            ff_dropout=ff_dropout, conv_dropout=conv_dropout,
        )
        self.buan = BUANBlock(
            dim=dim, dim_head=dim_head, heads=heads,
            attn_dropout=attn_dropout, ff_mult=ff_mult, ff_dropout=ff_dropout,
        )
        self.fusion = fusion
        if fusion == "concat":
            self.fusion_proj = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ccan_out = self.ccan(x)
        buan_out = self.buan(x)
        if self.fusion == "concat":
            return self.fusion_proj(torch.cat([ccan_out, buan_out], dim=-1))
        return ccan_out + buan_out

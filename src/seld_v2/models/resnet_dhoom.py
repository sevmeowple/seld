from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from seld_v2.models.components.resnet import resnet18_nopool
from seld_v2.models.components.conformer import ConformerBlock
from seld_v2.models.components.mask import subsequent_chunk_mask


class DHOOM(nn.Module):
    """Dual-path Hybrid Offline-Online Model for SELD.

    Left path: causal (streaming-capable)
    Right path: non-causal (full context)
    Dual heads: offline (left+right fused) and streaming (left-only)
    """

    def __init__(
        self,
        in_channel: int = 7,
        in_dim: int = 64,
        out_dim: int = 39,
        num_conformer_layers: int = 4,
        encoder_dim: int = 256,
        num_classes: int = 13,
        num_mhsa: int = 0,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim

        # Shared ResNet encoder (run twice: causal + noncausal)
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256  # after ResNet: 64//32*256 = 512
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )

        # Shared conformer layers (run twice per layer: right then left)
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=encoder_dim, dim_head=32, heads=8, ff_mult=2,
                conv_expansion_factor=2, conv_kernel_size=7,
                attn_dropout=dropout, ff_dropout=dropout, conv_dropout=dropout,
                att_context_size=[-1, -1],  # no cache-based chunking
            ) for _ in range(num_conformer_layers)
        ])

        # Block 1 fusion: concat + project
        self.first_block_proj = nn.Linear(2 * encoder_dim, encoder_dim)

        # Output norms
        self.offline_norm = nn.LayerNorm(encoder_dim)
        self.streaming_norm = nn.LayerNorm(encoder_dim)

        # Optional MHSA layers for offline path
        self.mhsa_layers = nn.ModuleList()
        self.mhsa_norms = nn.ModuleList()
        for _ in range(num_mhsa):
            self.mhsa_layers.append(
                nn.MultiheadAttention(encoder_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.mhsa_norms.append(nn.LayerNorm(encoder_dim))

        # Dual output heads
        self.t_pooling = nn.MaxPool1d(kernel_size=5)

        self.offline_sed = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, num_classes), nn.Sigmoid(),
        )
        self.offline_doa = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim), nn.Tanh(),
        )
        self.streaming_sed = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, num_classes), nn.Sigmoid(),
        )
        self.streaming_doa = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim), nn.Tanh(),
        )

    def _predict(self, features: torch.Tensor, sed_head: nn.Module, doa_head: nn.Module) -> torch.Tensor:
        out = self.t_pooling(features.permute(0, 2, 1)).permute(0, 2, 1)
        sed = sed_head(out)
        doa = doa_head(out)
        return torch.cat((sed, doa), dim=-1)

    def _build_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Lower-triangular causal mask: (1, 1, T, T)."""
        mask = subsequent_chunk_mask(T, 1, -1, device)  # chunk_size=1 → strict causal
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, T, F)

        # Shared ResNet: causal (left) and noncausal (right)
        left_2d = self.resnet(x)
        with self.resnet.noncausal_mode():
            right_2d = self.resnet(x)

        # Flatten (B, 256, T, 2) → (B, T, 512) → project
        def flatten_and_project(feat: torch.Tensor) -> torch.Tensor:
            B, C, T, W = feat.shape
            return self.input_projection(feat.permute(0, 2, 1, 3).reshape(B, T, C * W))

        left = flatten_and_project(left_2d)
        right = flatten_and_project(right_2d)

        # Build causal mask for left path
        T = left.shape[1]
        causal_mask = self._build_causal_mask(T, left.device)

        # Conformer dual-path
        for i, layer in enumerate(self.conformer_layers):
            merged = left + right

            # Right path: full context (no mask)
            right_out = layer(merged, mask=None)

            # Fuse for left input
            if i == 0:
                left_in = self.first_block_proj(torch.cat([merged, right_out], dim=-1))
            else:
                left_in = merged + right_out

            # Left path: causal mask
            left_out = layer(left_in, mask=causal_mask)

            right = right_out
            left = left_out

        # Output features
        offline_feat = self.offline_norm(left + right)
        streaming_feat = self.streaming_norm(left)

        # Optional MHSA for offline path
        for mhsa, norm in zip(self.mhsa_layers, self.mhsa_norms):
            residual = offline_feat
            attn_out, _ = mhsa(offline_feat, offline_feat, offline_feat)
            offline_feat = norm(attn_out + residual)

        # Dual predictions
        offline_pred = self._predict(offline_feat, self.offline_sed, self.offline_doa)
        streaming_pred = self._predict(streaming_feat, self.streaming_sed, self.streaming_doa)

        return offline_pred, streaming_pred

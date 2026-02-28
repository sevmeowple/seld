from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from seld_v2.models.components.resnet import resnet18_nopool
from seld_v2.models.components.hoom import HOOMLayer


class ResnetHoom(nn.Module):
    """ResNet-HOOM for SELD (offline only).

    Replaces Conformer blocks with HOOM layers (CCAN + BUAN dual-branch).
    Output shape is identical to ResnetConformer.
    """

    def __init__(
        self,
        in_channel: int,
        in_dim: int,
        out_dim: int,
        num_hoom_layers: int = 8,
        encoder_dim: int = 256,
        num_classes: int = 13,
        hoom_fusion: str = "concat",
    ):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256

        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        self.hoom_layers = nn.ModuleList([
            HOOMLayer(
                dim=encoder_dim, dim_head=32, heads=8, ff_mult=2,
                conv_expansion_factor=2, conv_kernel_size=7,
                attn_dropout=0.1, ff_dropout=0.1, conv_dropout=0.1,
                fusion=hoom_fusion,
            ) for _ in range(num_hoom_layers)
        ])
        self.after_norm = nn.LayerNorm(encoder_dim)
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        self.sed_out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, num_classes),
            nn.Sigmoid(),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim),
            nn.Tanh(),
        )

    def _predict(self, encoder_out: torch.Tensor) -> torch.Tensor:
        outputs = self.t_pooling(encoder_out.permute(0, 2, 1)).permute(0, 2, 1)
        sed = self.sed_out_layer(outputs)
        doa = self.out_layer(outputs)
        return torch.cat((sed, doa), dim=-1)

    def forward(
        self, x: torch.Tensor, return_encoder_outputs: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.resnet(x)
        N, C, T, W = conv_out.shape
        conv_out = conv_out.permute(0, 2, 1, 3).reshape(N, T, C * W)
        out = self.input_projection(conv_out)

        for layer in self.hoom_layers:
            out = layer(out)
        out = self.after_norm(out)

        pred = self._predict(out)
        if return_encoder_outputs:
            return pred, out
        return pred

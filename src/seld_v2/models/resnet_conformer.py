from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from seld_v2.models.components.resnet import resnet18_nopool
from seld_v2.models.components.conformer import ConformerBlock
from seld_v2.models.components.mask import add_optional_chunk_mask


class ResnetConformer(nn.Module):
    """Unified ResNet-Conformer for SELD.

    Merges the following original classes:
        - ResnetConformer_sed_doa_nopool (cache_resnet_conformer.py)
        - ResnetConformer_sed_doa_nopool_TS (teacher embedded → now external)
        - ResnetConformer_sed_doa_nopool_TS_after_conformer (same)
        - ResnetConformer_sed_doa_nopool_original (resnet_conformer_audio.py)
        - ResnetConformer_sed_doa_nopool_return_conformer_outputs (same)
        - ResnetConformer_sed_doa_nopool_Mixup (same)
    """

    def __init__(
        self,
        in_channel: int,
        in_dim: int,
        out_dim: int,
        att_context_size: List[int] = [100, 49],
        num_conformer_layer: int = 8,
        encoder_dim: int = 256,
        num_classes: int = 13,
        use_dynamic_chunk: bool = True,
        chunk_candidates: Optional[List[int]] = None,
    ):
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        self.encoder_dim = encoder_dim
        self.in_ch = in_channel
        self.in_dim = in_dim
        self.att_context_size = att_context_size
        self.cache_past_len = att_context_size[0]
        self.use_dynamic_chunk = use_dynamic_chunk
        self.chunk_candidates = chunk_candidates

        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=encoder_dim, dim_head=32, heads=8, ff_mult=2,
                conv_expansion_factor=2, conv_kernel_size=7,
                attn_dropout=0.1, ff_dropout=0.1, conv_dropout=0.1,
                att_context_size=att_context_size,
            ) for _ in range(num_conformer_layer)
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

    def _predict(self, conformer_outputs: torch.Tensor) -> torch.Tensor:
        outputs = self.t_pooling(conformer_outputs.permute(0, 2, 1)).permute(0, 2, 1)
        sed = self.sed_out_layer(outputs)
        doa = self.out_layer(outputs)
        return torch.cat((sed, doa), dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        resnet_cache: Optional[Tuple] = None,
        conformer_cache: Optional[List] = None,
        decoding_chunk_size: Optional[int] = None,
        return_conformer_outputs: bool = False,
    ) -> torch.Tensor | Tuple:
        is_streaming = resnet_cache is not None or conformer_cache is not None

        if not is_streaming:
            return self._forward_offline(x, decoding_chunk_size, return_conformer_outputs)
        return self._forward_streaming(x, resnet_cache, conformer_cache, return_conformer_outputs)

    def _forward_offline(
        self, x: torch.Tensor, decoding_chunk_size: Optional[int],
        return_conformer_outputs: bool,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        conv_out = self.resnet(x)
        N, C, T, W = conv_out.shape
        conv_out = conv_out.permute(0, 2, 1, 3).reshape(N, T, C * W)
        conformer_out = self.input_projection(conv_out)

        # build chunk mask
        chunk_size = self.att_context_size[1] + 1
        if decoding_chunk_size is None:
            active_chunk = 0 if self.training else chunk_size
        else:
            active_chunk = decoding_chunk_size
        # att_context_size[1] < 0 means full context, skip chunk mask
        if chunk_size <= 0 and active_chunk <= 0:
            chunk_mask = None
        else:
            masks = torch.ones(N, 1, T, dtype=torch.bool, device=x.device)
            num_left = self.att_context_size[0] // chunk_size if chunk_size > 0 else -1
            chunk_mask = add_optional_chunk_mask(
                conformer_out, masks,
                use_dynamic_chunk=self.use_dynamic_chunk, use_dynamic_left_chunk=True,
                decoding_chunk_size=active_chunk,
                static_chunk_size=max(chunk_size, 0), num_decoding_left_chunks=num_left,
                chunk_candidates=self.chunk_candidates,
            )
        if chunk_mask is not None:
            chunk_mask = chunk_mask.unsqueeze(1)

        for layer in self.conformer_layers:
            conformer_out = layer(conformer_out, mask=chunk_mask)
        conformer_out = self.after_norm(conformer_out)

        pred = self._predict(conformer_out)
        if return_conformer_outputs:
            return pred, conformer_out
        return pred

    def _forward_streaming(
        self, x: torch.Tensor, resnet_cache: Optional[Tuple],
        conformer_cache: Optional[List], return_conformer_outputs: bool,
    ) -> Tuple:
        conv_out, next_resnet_cache = self.resnet(x, resnet_cache)
        N, C, T, W = conv_out.shape
        conv_out = conv_out.permute(0, 2, 1, 3).reshape(N, T, C * W)
        conformer_out = self.input_projection(conv_out)

        next_layer_caches = []
        for i, layer in enumerate(self.conformer_layers):
            conformer_out, next_cache = layer(
                conformer_out,
                cache=conformer_cache[i] if conformer_cache else None,
            )
            next_layer_caches.append(next_cache)
        conformer_out = self.after_norm(conformer_out)

        pred = self._predict(conformer_out)
        caches = (next_resnet_cache, next_layer_caches)
        if return_conformer_outputs:
            return pred, conformer_out, caches
        return pred, caches

    def get_initial_cache_resnet(self, batch_size: int = 1) -> Tuple:
        device = next(self.parameters()).device
        conv1_cache = torch.zeros(batch_size, self.in_ch, 2, self.in_dim, device=device)
        channels = [(24, 24), (48, 48), (96, 96), (192, 192)]
        features = [64, 16, 4, 2]
        layer_caches = []
        for layer_idx, ((in_ch, out_ch), feat_dim) in enumerate(zip(channels, features)):
            current = []
            for block_idx in range(2):
                c1_ch = channels[layer_idx - 1][1] if block_idx == 0 and layer_idx > 0 else in_ch
                current.extend([
                    torch.zeros(batch_size, c1_ch, 2, feat_dim, device=device),
                    torch.zeros(batch_size, out_ch, 2, feat_dim, device=device),
                ])
            layer_caches.append(current)
        return (conv1_cache, layer_caches)

    def get_initial_cache_conformer(self, batch_size: int = 1) -> List:
        device = next(self.parameters()).device
        caches = []
        for _ in self.conformer_layers:
            attn_cache = torch.zeros(batch_size, self.cache_past_len, self.encoder_dim, device=device)
            conv_cache = torch.zeros(batch_size, 2 * self.encoder_dim, 6, device=device)
            caches.append((attn_cache, conv_cache))
        return caches

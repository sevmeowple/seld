"""ResNet-Conformer-DHOOM: Dual-path Hybrid Offline-Online Model with Layer-wise Fusion.

Architecture:
- ResNet18: Independent online/offline paths with BasicBlock-level interaction
- Conformer: Independent online/offline layers with layer-wise alternating fusion
- Unified interface: forward(x) for joint training, forward(x, cache) for streaming inference

Reference: Based on ResnetConformer_sed_doa_nopool_dual_online_offline_TS
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from seld_v2.models.components.resnet import resnet18_nopool
from seld_v2.models.components.conformer import ConformerBlock
from seld_v2.models.components.mask import add_optional_chunk_mask, subsequent_chunk_mask


class ResNetConformerDHOOM(nn.Module):
    """ResNet-Conformer-DHOOM with progressive layer-wise fusion.

    Phase 1: Conformer layer-wise interaction (implemented)
    Phase 2: ResNet BasicBlock-level interaction (implemented)

    Dual inference modes:
    - model(x) -> offline_pred (training, dual-path with fusion)
    - model(x, resnet_cache, conformer_cache) -> pred, caches (streaming inference)
    """

    def __init__(
        self,
        in_channel: int = 7,
        in_dim: int = 64,
        out_dim: int = 39,
        num_conformer_layers: int = 8,
        encoder_dim: int = 256,
        num_classes: int = 13,
        num_mhsa: int = 0,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_hidden_distill: bool = False,  # Disabled as per requirement
        att_context_size: List[int] | None = None,
        use_dynamic_chunk: bool = True,
        chunk_candidates: List[int] | None = None,
        sample_chunks_from_candidates: bool = False,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_conformer_layers = num_conformer_layers
        self.use_hidden_distill = use_hidden_distill

        # U2-style dynamic chunk parameters
        self.att_context_size = att_context_size if att_context_size is not None else [100, 49]
        self.use_dynamic_chunk = use_dynamic_chunk
        self.chunk_candidates = chunk_candidates
        self.sample_chunks_from_candidates = sample_chunks_from_candidates
        self.cache_past_len = self.att_context_size[0] if self.att_context_size[0] > 0 else 100

        # =====================================================================
        # Phase 2: Independent ResNet paths with BasicBlock interaction
        # =====================================================================
        self.resnet = resnet18_nopool(in_channel=in_channel)  # Offline (non-causal)
        self.online_resnet = resnet18_nopool(in_channel=in_channel)  # Online (causal)

        embedding_dim = in_dim // 32 * 256  # after ResNet: 64//32*256 = 512
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )
        self.online_input_projection = nn.Sequential(
            nn.Linear(embedding_dim, encoder_dim),
            nn.Dropout(p=0.05),
        )

        # =====================================================================
        # Phase 1: Independent Conformer layers with layer-wise fusion
        # =====================================================================
        # Offline (non-causal) Conformer layers
        self.offline_conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=encoder_dim, dim_head=32, heads=8, ff_mult=2,
                conv_expansion_factor=2, conv_kernel_size=7,
                attn_dropout=dropout, ff_dropout=dropout, conv_dropout=dropout,
                att_context_size=[-1, -1],  # no cache-based chunking for offline
            ) for _ in range(num_conformer_layers)
        ])

        # Online (causal) Conformer layers with U2-style chunking
        self.online_conformer_layers = nn.ModuleList([
            ConformerBlock(
                dim=encoder_dim, dim_head=32, heads=8, ff_mult=2,
                conv_expansion_factor=2, conv_kernel_size=7,
                attn_dropout=dropout, ff_dropout=dropout, conv_dropout=dropout,
                att_context_size=self.att_context_size,  # U2-style chunking
            ) for _ in range(num_conformer_layers)
        ])

        # Output norms
        self.offline_norm = nn.LayerNorm(encoder_dim)
        self.online_norm = nn.LayerNorm(encoder_dim)

        # Optional MHSA layers for offline path (from DHOOM)
        self.mhsa_layers = nn.ModuleList()
        self.mhsa_norms = nn.ModuleList()
        for _ in range(num_mhsa):
            self.mhsa_layers.append(
                nn.MultiheadAttention(encoder_dim, num_heads, dropout=dropout, batch_first=True)
            )
            self.mhsa_norms.append(nn.LayerNorm(encoder_dim))

        # Output heads (shared for simplicity, matching reference code pattern)
        self.t_pooling = nn.MaxPool1d(kernel_size=5)

        # Single output head (reference style: offline model provides the main output)
        self.sed_out = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, num_classes), nn.Sigmoid(),
        )
        self.doa_out = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim), nn.Tanh(),
        )

        # Online (streaming) output heads for dual-head training
        self.online_sed_out = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, num_classes), nn.Sigmoid(),
        )
        self.online_doa_out = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim), nn.LeakyReLU(),
            nn.Linear(encoder_dim, out_dim), nn.Tanh(),
        )

    def _predict(self, features: torch.Tensor, sed_head: nn.Module, doa_head: nn.Module) -> torch.Tensor:
        """Apply temporal pooling and output heads."""
        out = self.t_pooling(features.permute(0, 2, 1)).permute(0, 2, 1)
        sed = sed_head(out)
        doa = doa_head(out)
        return torch.cat((sed, doa), dim=-1)

    def _build_chunk_mask(
        self,
        T: int,
        device: torch.device,
        decoding_chunk_size: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Build chunk mask for online path using U2-style dynamic chunk.

        Args:
            T: Sequence length
            device: torch device
            decoding_chunk_size: Fixed chunk size for inference.
                None: Training mode (use dynamic chunk if enabled)
                0: Full context
                >0: Fixed chunk size

        Returns:
            Chunk mask of shape (1, 1, T, T) or None for full context.
        """
        chunk_size = self.att_context_size[1] + 1  # e.g., 49+1=50

        if decoding_chunk_size is None:
            active_chunk = 0 if self.training else chunk_size
        else:
            active_chunk = decoding_chunk_size

        # Full context: no mask
        if chunk_size <= 0 and active_chunk <= 0:
            return None

        masks = torch.ones(1, T, device=device, dtype=torch.bool)
        num_left = self.cache_past_len // chunk_size if chunk_size > 0 else -1

        chunk_mask = add_optional_chunk_mask(
            xs=torch.zeros(1, T, self.encoder_dim, device=device),
            masks=masks,
            use_dynamic_chunk=self.use_dynamic_chunk,
            use_dynamic_left_chunk=True,
            decoding_chunk_size=active_chunk,
            static_chunk_size=max(chunk_size, 0),
            num_decoding_left_chunks=num_left,
            chunk_candidates=self.chunk_candidates,
            sample_from_candidates=self.sample_chunks_from_candidates,
        )

        if chunk_mask is None or chunk_mask is masks:
            return None
        return chunk_mask.unsqueeze(1)  # (1, 1, T, T) for attention heads

    def forward(
        self,
        x: torch.Tensor,
        resnet_cache: Optional[Tuple] = None,
        conformer_cache: Optional[List] = None,
        decoding_chunk_size: Optional[int] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple]:
        """Dual-path forward with layer-wise fusion.

        Args:
            x: Input tensor (B, C, T, F)
            resnet_cache: Optional cache for streaming ResNet inference
            conformer_cache: Optional cache for streaming Conformer inference
            decoding_chunk_size: Fixed chunk size for online path during inference.
                None: Training mode (use dynamic chunk if enabled)
                0 or -1: Full context (offline-like)
                >0: Fixed chunk size

        Returns:
            If cache is None: (offline_pred, online_pred) tuple (training mode)
            If cache is provided: (prediction, (next_resnet_cache, next_conformer_cache))
        """
        # =====================================================================
        # Streaming inference mode (cache provided)
        # =====================================================================
        if resnet_cache is not None or conformer_cache is not None:
            return self._forward_streaming(x, resnet_cache, conformer_cache)

        # =====================================================================
        # Joint training mode (no cache)
        # Phase 1: Conformer layer-wise interaction with shared ResNet
        # =====================================================================
        return self._forward_offline_online_joint(x, decoding_chunk_size)

    def _forward_offline_online_joint(
        self,
        x: torch.Tensor,
        decoding_chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Joint training: both offline and online paths with layer-wise fusion.

        Phase 2: ResNet BasicBlock-level interaction (implemented)
        Phase 1: Conformer layer-wise interaction (implemented)

        Args:
            x: Input tensor (B, C, T, F)
            decoding_chunk_size: Fixed chunk size for inference.
                None: Training mode (use dynamic chunk if enabled)
                0: Full context (offline-like)
                >0: Fixed chunk size
        """
        B, C, T, F = x.shape

        # -----------------------------------------------------------------
        # Phase 2: ResNet with BasicBlock-level interaction
        # Reference pattern from cache_resnet_conformer_dual.py
        # Each BasicBlock alternates: offline(offline + online) -> online(online + offline)
        # -----------------------------------------------------------------

        # Helper function for applying normalization and activation
        def apply_ln_relu(resnet_module, feat: torch.Tensor) -> torch.Tensor:
            """Apply LayerNorm and ReLU for ResNet."""
            N, C_feat, T_feat, W = feat.shape
            feat = feat.permute(0, 2, 1, 3).reshape(N, T_feat, -1)
            feat = resnet_module.ln1(feat)
            feat = feat.reshape(N, T_feat, C_feat, W).permute(0, 2, 1, 3)
            return resnet_module.relu(feat)

        # Conv1: Both paths use LayerNorm (ResNet_nopool uses LayerNorm)
        # Online (causal) path
        online_feat = self.online_resnet.conv1(x)
        online_feat = apply_ln_relu(self.online_resnet, online_feat)

        # Offline (non-causal) path
        offline_feat = self.resnet.conv1(x)
        offline_feat = apply_ln_relu(self.resnet, offline_feat)

        # Layer 1: 2 BasicBlocks (24 channels, 64 freq dim) -> maxpool1
        # Reference pattern: both paths use same previous features
        # Block 0
        offline_prev = offline_feat
        online_prev = online_feat
        offline_feat = self.resnet.layer1[0](offline_prev + online_prev)
        online_feat = self.online_resnet.layer1[0](online_prev + offline_prev)
        # Block 1
        offline_prev = offline_feat
        online_prev = online_feat
        offline_feat = self.resnet.layer1[1](offline_prev + online_prev)
        online_feat = self.online_resnet.layer1[1](online_prev + offline_prev)
        # MaxPool
        offline_feat = self.resnet.maxpool1(offline_feat)
        online_feat = self.online_resnet.maxpool1(online_feat)

        # Layer 2: 2 BasicBlocks (48 channels, 16 freq dim) -> maxpool2
        # Reference pattern: store previous features before update
        offline_prev = offline_feat
        online_prev = online_feat
        # Block 0
        offline_feat = self.resnet.layer2[0](offline_prev + online_prev)
        online_feat = self.online_resnet.layer2[0](online_prev + offline_prev)
        # Block 1
        offline_prev = offline_feat
        online_prev = online_feat
        offline_feat = self.resnet.layer2[1](offline_prev + online_prev)
        online_feat = self.online_resnet.layer2[1](online_prev + offline_prev)
        # MaxPool
        offline_feat = self.resnet.maxpool2(offline_feat)
        online_feat = self.online_resnet.maxpool2(online_feat)

        # Layer 3: 2 BasicBlocks (96 channels, 4 freq dim) -> maxpool3
        offline_prev = offline_feat
        online_prev = online_feat
        # Block 0
        offline_feat = self.resnet.layer3[0](offline_prev + online_prev)
        online_feat = self.online_resnet.layer3[0](online_prev + offline_prev)
        # Block 1
        offline_prev = offline_feat
        online_prev = online_feat
        offline_feat = self.resnet.layer3[1](offline_prev + online_prev)
        online_feat = self.online_resnet.layer3[1](online_prev + offline_prev)
        # MaxPool
        offline_feat = self.resnet.maxpool3(offline_feat)
        online_feat = self.online_resnet.maxpool3(online_feat)

        # Layer 4: 2 BasicBlocks (192 channels, 2 freq dim) -> conv5
        offline_prev = offline_feat
        online_prev = online_feat
        # Block 0
        offline_feat = self.resnet.layer4[0](offline_prev + online_prev)
        online_feat = self.online_resnet.layer4[0](online_prev + offline_prev)
        # Block 1
        offline_prev = offline_feat
        online_prev = online_feat
        offline_feat = self.resnet.layer4[1](offline_prev + online_prev)
        online_feat = self.online_resnet.layer4[1](online_prev + offline_prev)

        # Conv5 (1x1 conv to 256 channels)
        offline_feat = self.resnet.conv5(offline_feat)
        online_feat = self.online_resnet.conv5(online_feat)

        # Flatten and project: (B, 256, T, 2) -> (B, T, 512) -> (B, T, 256)
        def flatten_and_project(feat_2d: torch.Tensor, projection: nn.Module) -> torch.Tensor:
            B_feat, C_feat, T_feat, W = feat_2d.shape
            return projection(feat_2d.permute(0, 2, 1, 3).reshape(B_feat, T_feat, C_feat * W))

        online_feat = flatten_and_project(online_feat, self.input_projection)   # (B, T, 256)
        offline_feat = flatten_and_project(offline_feat, self.input_projection)  # (B, T, 256)

        # -----------------------------------------------------------------
        # Conformer with layer-wise alternating fusion
        # Reference pattern:
        #   offline = offline_layer[i](offline + online)
        #   online = online_layer[i](online + offline)
        # -----------------------------------------------------------------
        # Build U2-style chunk mask for online path
        chunk_mask = self._build_chunk_mask(T, x.device, decoding_chunk_size)

        for i in range(self.num_conformer_layers):
            # Offline path: use combined features (full context, no mask)
            offline_feat = self.offline_conformer_layers[i](
                offline_feat + online_feat, mask=None
            )

            # Online path: use offline output for fusion (U2-style chunk mask)
            online_feat = self.online_conformer_layers[i](
                online_feat + offline_feat, mask=chunk_mask
            )

        # Normalize outputs
        offline_feat = self.offline_norm(offline_feat)
        online_feat = self.online_norm(online_feat)

        # Optional MHSA for offline path
        for mhsa, norm in zip(self.mhsa_layers, self.mhsa_norms):
            residual = offline_feat
            attn_out, _ = mhsa(offline_feat, offline_feat, offline_feat)
            offline_feat = norm(attn_out + residual)

        # -----------------------------------------------------------------
        # Output predictions
        # Reference style: return offline prediction as main output
        # For dual-head training, we need both predictions
        # -----------------------------------------------------------------
        offline_pred = self._predict(offline_feat, self.sed_out, self.doa_out)
        online_pred = self._predict(online_feat, self.online_sed_out, self.online_doa_out)

        # Return format matching reference: single tensor or list
        # For training, we return offline_pred (main output)
        # The training script can call model.online(x) separately for online output
        # But for dual-head loss, we need both here
        # Following DHOOM pattern: return (offline_pred, online_pred)
        return offline_pred, online_pred

    def _forward_streaming(
        self,
        x: torch.Tensor,
        resnet_cache: Optional[Tuple],
        conformer_cache: Optional[List],
    ) -> Tuple[torch.Tensor, Tuple]:
        """Streaming inference: online path only with cache."""
        # ResNet with cache (using online_resnet for streaming)
        if resnet_cache is not None:
            conv_out, next_resnet_cache = self.online_resnet(x, resnet_cache)
        else:
            conv_out = self.online_resnet(x)
            next_resnet_cache = None

        B, C, T, W = conv_out.shape
        conv_out = conv_out.permute(0, 2, 1, 3).reshape(B, T, C * W)
        feat = self.input_projection(conv_out)

        # Conformer with cache
        next_conformer_caches = []
        if conformer_cache is not None:
            for i, layer in enumerate(self.online_conformer_layers):
                cache = conformer_cache[i] if conformer_cache else None
                feat, next_cache = layer(feat, mask=None, cache=cache)
                next_conformer_caches.append(next_cache)
        else:
            for layer in self.online_conformer_layers:
                feat = layer(feat, mask=None)

        # Output
        feat = self.online_norm(feat)
        pred = self._predict(feat, self.online_sed_out, self.online_doa_out)

        return pred, (next_resnet_cache, next_conformer_caches)

    def get_initial_cache_resnet(self, batch_size: int = 1) -> Tuple:
        """Initialize ResNet cache for streaming inference."""
        device = next(self.parameters()).device

        conv1_cache = torch.zeros(batch_size, self.online_resnet.conv1.conv.in_channels, 2, 64, device=device)

        layer_caches = []
        channels = [(24, 24), (48, 48), (96, 96), (192, 192)]
        features = [64, 16, 4, 2]

        for layer_idx, ((in_ch, out_ch), feat_dim) in enumerate(zip(channels, features)):
            current_layer_caches = []
            num_blocks = 2

            for block_idx in range(num_blocks):
                if block_idx == 0 and layer_idx > 0:
                    prev_ch = channels[layer_idx - 1][1]
                    cache1 = torch.zeros(batch_size, prev_ch, 2, feat_dim, device=device)
                else:
                    cache1 = torch.zeros(batch_size, in_ch, 2, feat_dim, device=device)
                cache2 = torch.zeros(batch_size, out_ch, 2, feat_dim, device=device)
                current_layer_caches.extend([cache1, cache2])

            layer_caches.append(current_layer_caches)

        return (conv1_cache, layer_caches)

    def get_initial_cache_conformer(self, batch_size: int = 1) -> List:
        """Initialize Conformer cache for streaming inference."""
        caches = []
        device = next(self.parameters()).device

        for layer in self.online_conformer_layers:
            # attention cache: [B, cache_len, D] - use cache_past_len from att_context_size
            attn_cache = torch.zeros(batch_size, self.cache_past_len, self.encoder_dim, device=device)
            # conv cache: [B, 2*D, kernel_size-1]
            conv_cache = torch.zeros(batch_size, 2 * self.encoder_dim, 6, device=device)
            caches.append((attn_cache, conv_cache))

        return caches

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
from torch import nn, einsum
from einops import rearrange

from seld_v2.models.common import (
    GLU, Swish, DepthWiseConv1dWithCache, Scale, PreNorm,
)


class CausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 dropout: float = 0., max_pos_emb: int = 512,
                 att_context_size: List[int] = [100, 49]):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.max_pos_emb = max_pos_emb

        self.to_q = nn.Linear(dim, inner_dim, bias=True)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=True)
        self.to_out = nn.Linear(inner_dim, dim)

        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        self.dropout = nn.Dropout(dropout)
        self.max_cache_len = att_context_size[0]
        self.att_context_size = att_context_size

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        input_len = x.shape[1]
        device, h = x.device, self.heads

        if cache is not None:
            full_context = torch.cat([cache, x], dim=1)
            next_cache = full_context[:, -self.max_cache_len:, :]
            if self.att_context_size[1] >= 0:
                max_context = self.att_context_size[1] + input_len
                context = full_context[:, -max_context:, :] if full_context.shape[1] > max_context else full_context
            else:
                context = full_context
        else:
            context = x

        context_len = context.shape[1]

        # Q from current input only; K/V from full context (WeNet style)
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # relative position encoding with Q/K length mismatch
        seq_q = torch.arange(input_len, device=device) + (context_len - input_len)
        seq_k = torch.arange(context_len, device=device)
        dist = rearrange(seq_q, 'i -> i ()') - rearrange(seq_k, 'j -> () j')
        dist = dist.clamp(-self.max_pos_emb, self.max_pos_emb) + self.max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist)
        pos_attn = einsum('b h n d, n j d -> b h n j', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if mask is not None:
            dots.masked_fill_(~mask, float('-inf'))

        attn = dots.softmax(dim=-1)
        if mask is not None:
            attn = attn.masked_fill(~mask, 0.0)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        if cache is not None:
            return out, next_cache
        return out


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    def __init__(self, dim: int, expansion_factor: int = 2,
                 kernel_size: int = 31, dropout: float = 0.):
        super().__init__()
        inner_dim = dim * expansion_factor
        padding = (kernel_size - 1, 0)
        self.conv1 = nn.Conv1d(dim, inner_dim * 2, 1)
        self.glu = GLU(dim=1)
        self.depth_conv = DepthWiseConv1dWithCache(
            inner_dim, inner_dim, kernel_size=kernel_size, padding=padding,
        )
        self.bn = nn.BatchNorm1d(inner_dim)
        self.swish = Swish()
        self.conv2 = nn.Conv1d(inner_dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        x = rearrange(x, 'b n c -> b c n')
        x = self.conv1(x)
        x = self.glu(x)
        if cache is None:
            x = self.depth_conv(x)
        else:
            x, next_cache = self.depth_conv(x, cache)
        x = self.bn(x)
        x = self.swish(x)
        x = self.conv2(x)
        x = rearrange(x, 'b c n -> b n c')
        x = self.dropout(x)
        if cache is None:
            return x
        return x, next_cache


class ConformerBlock(nn.Module):
    def __init__(
        self, *, dim: int, dim_head: int = 64, heads: int = 8,
        ff_mult: int = 4, conv_expansion_factor: int = 2,
        conv_kernel_size: int = 31, attn_dropout: float = 0.,
        ff_dropout: float = 0., conv_dropout: float = 0.,
        att_context_size: List[int] = [100, 49],
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = CausalAttention(
            dim=dim, dim_head=dim_head, heads=heads,
            dropout=attn_dropout, att_context_size=att_context_size,
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

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if cache is None:
            x = self.ff1(x) + x
            x = self.attn(x, mask=mask) + x
            x = self.conv(x) + x
            x = self.ff2(x) + x
            return self.post_norm(x)

        attn_cache, conv_cache = cache
        x = self.ff1(x) + x
        attn_out, next_attn_cache = self.attn(x, mask=mask, cache=attn_cache)
        x = attn_out + x
        conv_out, next_conv_cache = self.conv(x, cache=conv_cache)
        x = conv_out + x
        x = self.ff2(x) + x
        return self.post_norm(x), (next_attn_cache, next_conv_cache)

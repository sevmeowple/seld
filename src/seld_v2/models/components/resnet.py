from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2D(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        stride: int = 1, dilation: int = 1, groups: int = 1, bias: bool = True,
    ):
        super().__init__()
        padding = (0, kernel_size // 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.cache_len = kernel_size - 1

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    def forward(
        self, x: torch.Tensor, cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if cache is None:
            x = F.pad(x, (0, 0, self.cache_len, 0))
        else:
            x = torch.cat([cache, x], dim=2)

        output = self.conv(x)

        if cache is None:
            return output
        next_cache = x[:, :, -self.cache_len:, :]
        return output, next_cache


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> CausalConv2D:
    return CausalConv2D(in_planes, out_planes, kernel_size=3, stride=stride, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# channel -> feature_dim mapping for LayerNorm sizes
_LN_FEAT_DIM = {24: 64, 48: 16, 96: 4, 192: 2}


def _ln_size(planes: int) -> int:
    return planes * _LN_FEAT_DIM[planes]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: Optional[nn.Sequential] = None):
        super().__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.ln1 = nn.LayerNorm(_ln_size(planes))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = nn.LayerNorm(_ln_size(planes))
        self.downsample = downsample
        self.stride = stride

    def _apply_ln(self, x: torch.Tensor, ln: nn.LayerNorm) -> torch.Tensor:
        N, C, T, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(N, T, -1)
        x = ln(x)
        return x.reshape(N, T, C, W).permute(0, 2, 1, 3)

    def _apply_downsample(self, x: torch.Tensor) -> torch.Tensor:
        assert self.downsample is not None
        x = self.downsample[0](x)
        N, C, T, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(N, T, -1)
        x = self.downsample[1](x)
        return x.reshape(N, T, C, W).permute(0, 2, 1, 3)

    def forward(
        self, x: torch.Tensor, cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        identity = x
        if cache is None:
            out = self.conv1(x)
            out = self._apply_ln(out, self.ln1)
            out = self.relu(out)
            out = self.conv2(out)
            out = self._apply_ln(out, self.ln2)
            if self.downsample is not None:
                identity = self._apply_downsample(x)
            return self.relu(out + identity)

        cache1, cache2 = cache
        out, next_cache1 = self.conv1(x, cache1)
        out = self._apply_ln(out, self.ln1)
        out = self.relu(out)
        out, next_cache2 = self.conv2(out, cache2)
        out = self._apply_ln(out, self.ln2)
        if self.downsample is not None:
            identity = self._apply_downsample(x)
        return self.relu(out + identity), (next_cache1, next_cache2)


class ResNet_nopool(nn.Module):
    def __init__(self, block: type[BasicBlock], layers: List[int], in_channel: int = 17,
                 zero_init_residual: bool = False):
        super().__init__()
        self.inplanes = 24
        self.conv1 = CausalConv2D(in_channel, 24, kernel_size=3, stride=1, bias=False)
        self.ln1 = nn.LayerNorm(64 * 24)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 24, layers[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 4))
        self.layer2 = self._make_layer(block, 48, layers[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 4))
        self.layer3 = self._make_layer(block, 96, layers[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))
        self.layer4 = self._make_layer(block, 192, layers[3])
        self.conv5 = conv1x1(192, 256)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.ln2.weight, 0)

    def _make_layer(self, block: type[BasicBlock], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.LayerNorm(_ln_size(planes)),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _apply_ln(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, W = x.shape
        x = x.permute(0, 2, 1, 3).reshape(N, T, -1)
        x = self.ln1(x)
        return x.reshape(N, T, C, W).permute(0, 2, 1, 3)

    def forward(
        self, x: torch.Tensor, caches: Optional[Tuple[torch.Tensor, List]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, List]]:
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]

        if caches is None:
            x = self.conv1(x)
            x = self._apply_ln(x)
            x = self.relu(x)
            for i, layer in enumerate(layers):
                x = layer(x)
                if i < 3:
                    x = getattr(self, f'maxpool{i + 1}')(x)
            return self.conv5(x)

        conv1_cache, layer_caches = caches
        x, next_conv1_cache = self.conv1(x, conv1_cache)
        x = self._apply_ln(x)
        x = self.relu(x)

        next_layer_caches: List = []
        for i, layer in enumerate(layers):
            current_caches: List = []
            for j, block in enumerate(layer):
                idx = j * 2
                block_cache = (layer_caches[i][idx], layer_caches[i][idx + 1])
                x, next_block_cache = block(x, block_cache)
                current_caches.extend(next_block_cache)
            next_layer_caches.append(current_caches)
            if i < 3:
                x = getattr(self, f'maxpool{i + 1}')(x)
        return self.conv5(x), (next_conv1_cache, next_layer_caches)


def resnet18_nopool(**kwargs) -> ResNet_nopool:
    return ResNet_nopool(BasicBlock, [2, 2, 2, 2], **kwargs)

from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SED 损失函数
# ============================================================

def bce_sed_loss(sed_out: torch.Tensor, sed_label: torch.Tensor) -> torch.Tensor:
    """标准 BCE 损失"""
    return F.binary_cross_entropy(sed_out, sed_label)


def weighted_bce_sed_loss(class_weights: torch.Tensor) -> Callable:
    """带类别权重的 BCE 损失"""
    def fn(sed_out: torch.Tensor, sed_label: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy(sed_out, sed_label, reduction="none")
        return (loss * class_weights.to(loss.device)).mean()
    return fn


def kl_sed_loss(sed_out: torch.Tensor, sed_label: torch.Tensor) -> torch.Tensor:
    """KL 散度损失"""
    kl1 = (sed_label * torch.log(1e-7 + sed_label / (1e-7 + sed_out))).mean()
    kl2 = ((1 - sed_label) * torch.log(1e-7 + (1 - sed_label) / (1e-7 + 1 - sed_out))).mean()
    return kl1 + kl2


# ============================================================
# DOA 损失函数
# ============================================================

def single_mask_doa_loss(doa_out: torch.Tensor, doa_label: torch.Tensor, sed_mask: torch.Tensor) -> torch.Tensor:
    """只 mask 预测端"""
    return F.mse_loss(doa_out * sed_mask, doa_label)


def double_mask_doa_loss(doa_out: torch.Tensor, doa_label: torch.Tensor, sed_mask: torch.Tensor) -> torch.Tensor:
    """双边 mask"""
    return F.mse_loss(doa_out * sed_mask, doa_label * sed_mask)


# ============================================================
# 组合损失
# ============================================================

SedLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
DoaLossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class SedDoaLoss(nn.Module):
    """
    SED + DOA 组合损失，通过注入不同的 sed/doa 损失函数实现变体。

    原始类对应的参数组合:
        SedDoaLoss:          sed_loss_fn=bce_sed_loss,  doa_loss_fn=single_mask_doa_loss
        SedDoaLoss_SedClass: sed_loss_fn=weighted_bce_sed_loss(weights), doa_loss_fn=single_mask_doa_loss
        SedDoaKLLoss:        sed_loss_fn=kl_sed_loss,   doa_loss_fn=single_mask_doa_loss
        SedDoaKLLoss_2:      sed_loss_fn=kl_sed_loss,   doa_loss_fn=double_mask_doa_loss
        SedDoaKLLoss_3:      sed_loss_fn=kl_sed_loss,   doa_loss_fn=double_mask_doa_loss, binarize_label=True
    """

    def __init__(
        self,
        loss_weight: List[float] = [1.0, 10.0],
        sed_loss_fn: SedLossFn = bce_sed_loss,
        doa_loss_fn: DoaLossFn = single_mask_doa_loss,
        binarize_label: bool = False,
        num_classes: int = 13,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.sed_loss_fn = sed_loss_fn
        self.doa_loss_fn = doa_loss_fn
        self.binarize_label = binarize_label
        self.num_classes = num_classes

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = self.num_classes
        sed_out = output[:, :, :n]
        doa_out = output[:, :, n:]
        sed_label = target[:, :, :n]
        doa_label = target[:, :, n:n * 4]

        if self.binarize_label:
            sed_label = (sed_label > 0.5).float()

        sed_mask = sed_label.repeat(1, 1, 3)
        loss_sed = self.sed_loss_fn(sed_out, sed_label)
        loss_doa = self.doa_loss_fn(doa_out, doa_label, sed_mask)

        return self.loss_weight[0] * loss_sed + self.loss_weight[1] * loss_doa

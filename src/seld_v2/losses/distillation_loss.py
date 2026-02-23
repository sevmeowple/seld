from typing import Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from seld_v2.losses.sed_doa_loss import SedDoaLoss, kl_sed_loss, double_mask_doa_loss


# ============================================================
# 逐层距离函数
# ============================================================

def mse_distance(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(student, teacher)


def normalized_mse_distance(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """L2 归一化后计算 MSE"""
    t = teacher / (torch.norm(teacher, p=2) + 1e-6)
    s = student / (torch.norm(student, p=2) + 1e-6)
    return F.mse_loss(s, t)


def cosine_distance(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity"""
    cos_sim = F.cosine_similarity(
        student.view(student.size(0), -1),
        teacher.view(teacher.size(0), -1),
        dim=1,
    )
    return (1 - cos_sim).mean()


DistanceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ============================================================
# 逐层蒸馏损失 (隐藏层 + 注意力图)
# ============================================================

class LayerwiseDistillationLoss(nn.Module):
    """
    逐层蒸馏损失，统一处理隐藏层和注意力图的蒸馏。

    原始类对应的参数组合:
        HiddenStateMSELoss:          distance_fn=mse_distance, layer_weights=None
        HiddenStateMSELoss_weighted: distance_fn=mse_distance, layer_weights=[0,...,0,1]
        HiddenStateMSELoss_lastlayer_ts_not_equal:
                                     distance_fn=mse_distance, layer_weights=[0,...,0,1]
        HiddenStateMSELoss_norm:     distance_fn=normalized_mse_distance, layer_weights=None
        HiddenStateCosineLoss:       distance_fn=cosine_distance, layer_weights=[1,1,1,1,1,1,1,0.5]
        AttentionMapMSELoss:         distance_fn=mse_distance, layer_weights=None
        AttentionMapMSELoss_weighted:distance_fn=mse_distance, layer_weights=[0,...,0,1]
        SimpleAttentionDivergenceLoss:
                                     distance_fn=normalized_mse_distance, layer_weights=None
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        distance_fn: DistanceFn = mse_distance,
        layer_weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.distance_fn = distance_fn
        self.layer_weights = layer_weights

    def forward(
        self,
        student_states: List[torch.Tensor],
        teacher_states: List[torch.Tensor],
    ) -> torch.Tensor:
        num_layers = len(teacher_states)
        weights = self.layer_weights or [1.0] * num_layers
        total_loss = torch.tensor(0.0, device=teacher_states[0].device)
        for i in range(num_layers):
            total_loss = total_loss + self.distance_fn(student_states[i], teacher_states[i]) * weights[i]
        return (total_loss / sum(weights)) * self.loss_weight


# ============================================================
# 输出级蒸馏损失 (Semantic Representation Distillation)
# ============================================================

class SemanticRepresentationDistillationLoss_KLLoss_2(nn.Module):
    """
    语义表示蒸馏损失 - 基于 SedDoaKLLoss_2 的形式。
    内部组合 SedDoaLoss(kl + double_mask)，接口为 (student_output, teacher_output)。

    原始类: SemanticRepresentationDistillationLoss_KLLoss_2
    """

    def __init__(self, loss_weight: List[float] = [1.0, 10.0]):
        super().__init__()
        self._loss = SedDoaLoss(
            loss_weight=loss_weight,
            sed_loss_fn=kl_sed_loss,
            doa_loss_fn=double_mask_doa_loss,
        )

    def forward(self, student_output: torch.Tensor, teacher_output: torch.Tensor) -> torch.Tensor:
        return self._loss(student_output, teacher_output)


class SemanticRepresentationDistillationLoss(nn.Module):
    """
    语义表示蒸馏损失 (SRD)
    使用教师模型的分类器作为语义批评者来评估学生表示。

    原始类: SemanticRepresentationDistillationLoss
    """

    def __init__(self, loss_weight: float = 0.1, distance_type: str = "mse"):
        super().__init__()
        self.loss_weight = loss_weight
        self.distance_type = distance_type

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        teacher_classifier: nn.Module,
    ) -> torch.Tensor:
        with torch.no_grad():
            teacher_logits = teacher_classifier(teacher_features)

        student_cross_logits = teacher_classifier(student_features)

        if self.distance_type == "mse":
            loss = F.mse_loss(student_cross_logits, teacher_logits.detach())
        elif self.distance_type == "kl":
            teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
            student_probs = F.log_softmax(student_cross_logits, dim=-1)
            loss = F.kl_div(student_probs, teacher_probs, reduction="batchmean")
        else:
            raise ValueError(f"Unsupported distance type: {self.distance_type}")

        return loss * self.loss_weight

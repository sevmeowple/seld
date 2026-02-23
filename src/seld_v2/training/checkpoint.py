from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """早停管理"""

    def __init__(self, patience: int = 40):
        self.patience = patience
        self.counter = 0
        self.best_score = float("inf")
        self.best_epoch = 0
        self.best_checkpoint = ""

    def step(self, score: float, epoch: int, checkpoint_path: str) -> bool:
        """返回 True 表示应该停止训练"""
        if score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.best_checkpoint = checkpoint_path
            self.counter = 0
            logger.info("New best model found SELD score: %.4f", score)
        else:
            self.counter += 1
            logger.info(
                "No improvement for %d epochs. Best: %.4f",
                self.counter, self.best_score,
            )
        return self.counter >= self.patience


def save_checkpoint(
    model: torch.nn.Module,
    output_dir: str | Path,
    epoch: int,
    step: int,
) -> str:
    """保存模型检查点，返回保存路径"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"checkpoint_epoch{epoch}_step{step}.pth")
    torch.save(model.state_dict(), path)
    logger.info("save checkpoint: %s", path)
    return path


def load_checkpoint(model: torch.nn.Module
    , path: str | Path) -> None:
    """加载模型检查点"""
    model.load_state_dict(torch.load(path))
    logger.info("loaded checkpoint: %s", path)

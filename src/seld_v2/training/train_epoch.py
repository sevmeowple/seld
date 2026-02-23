from __future__ import annotations

import logging
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    step_count: int,
    total_steps: int,
    epoch: int,
    log_interval: int = 100,
) -> Dict:
    """
    训练一个 epoch。

    Returns:
        dict with keys: train_loss, step_count, stop_training, train_time
    """
    model.train()
    losses = []
    stop_training = False
    start_time = time.time()

    for data in dataloader:
        input = data["input"].to(device)
        target = data["target"].to(device)
        optimizer.zero_grad()
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        step_count += 1

        if step_count % log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "epoch: %d, step: %d/%d, lr:%.6f, train_loss:%.4f",
                epoch, step_count, total_steps, lr, loss.item(),
            )

        if step_count >= total_steps:
            stop_training = True
            logger.info("Reached maximum number of steps")
            break

    torch.cuda.empty_cache()

    return {
        "train_loss": float(np.mean(losses)) if losses else 0.0,
        "step_count": step_count,
        "stop_training": stop_training,
        "train_time": time.time() - start_time,
    }

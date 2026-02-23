from __future__ import annotations

import logging
import os
import time
from typing import Dict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from seld_v2.metrics.result_collector import SedDoaResultCollector

logger = logging.getLogger(__name__)


def write_output_format_file(output_file: str, output_dict: dict) -> None:
    """将预测结果写入 CSV 文件"""
    with open(output_file, "w") as f:
        for frame_ind in output_dict:
            for value in output_dict[frame_ind]:
                if len(value) == 4:
                    f.write("{},{},{},{},{},{}\n".format(
                        int(frame_ind) + 1, int(value[0]), 0,
                        float(value[1]), float(value[2]), float(value[3]),
                    ))
                else:
                    f.write("{},{},{},{},{}\n".format(
                        int(frame_ind) + 1, int(value[0]), 0,
                        int(value[1]), int(value[2]),
                    ))


def eval_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    collector: SedDoaResultCollector,
    device: torch.device,
) -> Dict:
    """
    验证一个 epoch。

    Returns:
        dict with keys: test_loss, output_dict, eval_time
    """
    model.eval()
    losses = []
    start_time = time.time()

    for data in dataloader:
        input = data["input"].to(device)
        target = data["target"].to(device)
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)
            losses.append(loss.item())
        collector.add_batch(data["wav_names"], output)

    return {
        "test_loss": float(np.mean(losses)) if losses else 0.0,
        "output_dict": collector.get_result(),
        "eval_time": time.time() - start_time,
    }


def save_and_evaluate(
    output_dict: dict,
    output_dir: str | Path,
    ref_files_dir: str | Path,
) -> Dict:
    """
    保存 CSV 并计算 SELD 指标。

    Returns:
        dict with keys: ER, F, LE, LR, seld_scr, classwise_results
    """
    from seld.utils.feature.compute_seld_results import ComputeSELDResults

    os.makedirs(output_dir, exist_ok=True)
    for csv_name, perfile_dict in output_dict.items():
        path = os.path.join(output_dir, f"{csv_name}.csv")
        write_output_format_file(path, perfile_dict)

    score_obj = ComputeSELDResults(ref_files_folder=ref_files_dir)
    ER, F, LE, LR, seld_scr, classwise = score_obj.get_SELD_Results(output_dir)

    logger.info("ER/F/LE/LR/SELD: %.4f/%.4f/%.4f/%.4f/%.4f", ER, F, LE, LR, seld_scr)

    return {
        "ER": ER, "F": F, "LE": LE, "LR": LR,
        "seld_scr": seld_scr, "classwise_results": classwise,
    }

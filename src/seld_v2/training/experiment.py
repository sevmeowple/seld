from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from pathlib import Path


class ExperimentDir:
    """自动创建带时间戳的实验目录，集中管理所有输出。"""

    def __init__(self, base_dir: str | Path = "experiments", name: str = ""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{name}" if name else timestamp

        root = Path(base_dir) / dir_name
        # 同名冲突自动加后缀
        if root.exists():
            i = 1
            while (Path(base_dir) / f"{dir_name}_{i}").exists():
                i += 1
            root = Path(base_dir) / f"{dir_name}_{i}"

        self.root = root
        self.log_dir = root / "logs"
        self.checkpoint_dir = root / "checkpoints"
        self.dcase_output_dir = root / "dcase_output"

        for d in [self.log_dir, self.checkpoint_dir, self.dcase_output_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def log_file(self, name: str = "train") -> Path:
        return self.log_dir / f"{name}.log"

    def setup_logging(self, log_name: str = "train") -> logging.Logger:
        logging.basicConfig(
            filename=str(self.log_file(log_name)), filemode="w",
            level=logging.INFO,
            format="%(levelname)s: %(asctime)s: %(message)s",
        )
        return logging.getLogger(__name__)

    def log_metrics(self, phase: str, config: dict, metrics: dict) -> None:
        """Append one row to experiments/metrics.csv with config + metrics."""
        csv_path = self.root.parent / "metrics.csv"
        row = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "exp_name": self.root.name,
            "phase": phase,
            **{f"cfg_{k}": v for k, v in config.items()},
            **metrics,
        }
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

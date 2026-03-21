from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import tomli
import tomli_w


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

    def log_metrics(
        self, phase: str, config: dict, metrics: dict, split_result: bool = True
    ) -> None:
        """Append or update an entry in experiments/metrics.toml.

        For train phase, replaces existing entry with same exp_name
        (keeps only the best). For test phase, always appends.

        Args:
            phase: "train" or "test"
            config: Configuration dictionary
            metrics: Metrics dictionary
            split_result: If True, also save to experiments/result/YYYYWxx/{exp_name}.toml
        """
        toml_path = self.root.parent / "metrics.toml"

        # Build classwise sub-dict from numpy array (5, N_classes)
        classwise_raw = metrics.get("classwise_results")
        classwise = {}
        if classwise_raw is not None:
            names = ["ER", "F", "LE", "LR", "seld_scr"]
            arr = np.asarray(classwise_raw)
            for i, name in enumerate(names):
                classwise[name] = arr[i].tolist()

        entry = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "exp_name": self.root.name,
            "phase": phase,
            "config": {k: _toml_safe(v) for k, v in config.items()},
            "metrics": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in metrics.items() if k != "classwise_results"
            },
        }
        if classwise:
            entry["classwise"] = classwise

        # Load existing
        data: dict = {"experiments": []}
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                data = tomli.load(f)

        # For train: replace previous entry with same exp_name
        if phase == "train":
            data["experiments"] = [
                e for e in data["experiments"]
                if not (e.get("exp_name") == entry["exp_name"] and e.get("phase") == "train")
            ]

        data["experiments"].append(entry)

        with open(toml_path, "wb") as f:
            tomli_w.dump(data, f)

        # Also save to split result directory if enabled
        if split_result:
            self._save_split_result(entry)

    def _get_week_folder(self, exp_name: str) -> str:
        """Extract week folder name from exp_name, e.g., 2026W08.

        Args:
            exp_name: Experiment name starting with YYYYMMDD

        Returns:
            Week folder string in format YYYYWxx
        """
        date_str = exp_name[:8]  # First 8 chars are YYYYMMDD
        dt = datetime.strptime(date_str, "%Y%m%d")
        iso = dt.isocalendar()
        return f"{iso.year}W{iso.week:02d}"

    def _save_split_result(self, entry: dict) -> None:
        """Save entry to experiments/result/YYYYWxx/{exp_name}.toml.

        Args:
            entry: The experiment entry to save
        """
        week_folder = self._get_week_folder(self.root.name)
        result_dir = self.root.parent / "result" / week_folder
        result_dir.mkdir(parents=True, exist_ok=True)

        single_file = result_dir / f"{self.root.name}.toml"
        with open(single_file, "wb") as f:
            tomli_w.dump(entry, f)


def _toml_safe(v):
    """Convert value to TOML-compatible type."""
    if isinstance(v, (list, tuple)):
        return [_toml_safe(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v

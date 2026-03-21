"""One-time migration: metrics.csv → metrics.toml, deduplicating train entries."""
import ast
import csv
import re
from pathlib import Path

import numpy as np
import tomli_w

CSV_PATH = Path("experiments/metrics.csv")
TOML_PATH = Path("experiments/metrics.toml")

METRIC_NAMES = ["ER", "F", "LE", "LR", "seld_scr"]
CONFIG_PREFIX = "cfg_"


def parse_classwise(raw: str) -> dict[str, list[float]] | None:
    if not raw or not raw.strip():
        return None
    # Normalize numpy repr to valid nested list
    s = raw.strip()
    s = re.sub(r"(\d)\s+([\d\-])", r"\1, \2", s)  # add commas between numbers
    s = re.sub(r"\]\s*\[", r"], [", s)  # add commas between rows
    s = re.sub(r"(\d\.)\s+", r"\1, ", s)  # trailing decimal
    try:
        arr = np.array(ast.literal_eval(s))
    except Exception:
        return None
    result = {}
    for i, name in enumerate(METRIC_NAMES):
        result[name] = [round(float(x), 6) for x in arr[i]]
    return result


def parse_config_value(v: str):
    """Try to parse config values to native types."""
    if not v:
        return v
    # Try list like [100, -1]
    try:
        parsed = ast.literal_eval(v)
        if isinstance(parsed, (list, tuple, int, float, bool)):
            return parsed
        return v
    except Exception:
        return v


def convert():
    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    experiments = []
    # Track best train entry per exp_name (last one = best seld_scr)
    train_best: dict[str, dict] = {}

    for row in rows:
        phase = row.get("phase", "")
        config = {}
        metrics = {}

        for k, v in row.items():
            if k in ("timestamp", "exp_name", "phase", "classwise_results"):
                continue
            if k.startswith(CONFIG_PREFIX):
                parsed = parse_config_value(v)
                if parsed != "":
                    config[k[len(CONFIG_PREFIX):]] = parsed
            elif k in METRIC_NAMES:
                try:
                    metrics[k] = round(float(v), 6)
                except (ValueError, TypeError):
                    metrics[k] = v

        classwise = parse_classwise(row.get("classwise_results", ""))

        entry = {
            "timestamp": row["timestamp"],
            "exp_name": row["exp_name"],
            "phase": phase,
            "config": config,
            "metrics": metrics,
        }
        if classwise:
            entry["classwise"] = classwise

        if phase == "train":
            # Keep only last (best) per exp_name
            train_best[row["exp_name"]] = entry
        else:
            experiments.append(entry)

    # Insert deduplicated train entries
    experiments.extend(train_best.values())
    # Sort by timestamp
    experiments.sort(key=lambda e: e["timestamp"])

    data = {"experiments": experiments}
    with open(TOML_PATH, "wb") as f:
        tomli_w.dump(data, f)

    print(f"Converted {len(rows)} CSV rows → {len(experiments)} TOML entries")
    print(f"  (deduplicated {len(rows) - len(experiments)} train entries)")
    print(f"Written to {TOML_PATH}")


if __name__ == "__main__":
    convert()

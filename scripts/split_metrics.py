#!/usr/bin/env python3
"""Split experiments/metrics.toml into separate files organized by week.

Each experiment is saved as a separate TOML file in experiments/result/YYYYWxx/.
The week folder is calculated from the experiment name (YYYYMMDD format).
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import tomli
import tomli_w


def get_week_folder(exp_name: str) -> str:
    """Extract week folder name from exp_name, e.g., 2026W08.

    Args:
        exp_name: Experiment name starting with YYYYMMDD (e.g., 20260221_171241_...)

    Returns:
        Week folder string in format YYYYWxx (e.g., 2026W08)
    """
    date_str = exp_name[:8]  # First 8 chars are YYYYMMDD
    dt = datetime.strptime(date_str, "%Y%m%d")
    iso = dt.isocalendar()
    return f"{iso.year}W{iso.week:02d}"


def split_metrics(
    metrics_path: Path,
    result_dir: Path,
    backup: bool = True,
    dry_run: bool = False,
) -> dict:
    """Split metrics.toml into separate files organized by week.

    Args:
        metrics_path: Path to metrics.toml
        result_dir: Directory to store split results
        backup: Whether to create a backup of original file
        dry_run: If True, only print what would be done without writing files

    Returns:
        Statistics dict with counts
    """
    if not metrics_path.exists():
        print(f"Error: {metrics_path} not found")
        return {"total": 0, "split": 0, "errors": 1}

    # Load metrics.toml
    with open(metrics_path, "rb") as f:
        data = tomli.load(f)

    experiments = data.get("experiments", [])
    total = len(experiments)

    print(f"Found {total} experiments in {metrics_path}")

    if total == 0:
        print("No experiments to split")
        return {"total": 0, "split": 0, "errors": 0}

    # Track created directories and processed files
    created_dirs: set[Path] = set()
    week_counts: dict[str, int] = {}

    for entry in experiments:
        exp_name = entry.get("exp_name", "")
        if not exp_name:
            print("Warning: Skipping entry without exp_name")
            continue

        week_folder = get_week_folder(exp_name)
        week_counts[week_folder] = week_counts.get(week_folder, 0) + 1

        exp_dir = result_dir / week_folder
        output_file = exp_dir / f"{exp_name}.toml"

        if dry_run:
            print(f"Would create: {output_file}")
            continue

        # Create directory if not exists
        if exp_dir not in created_dirs:
            exp_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(exp_dir)
            print(f"Created directory: {exp_dir}")

        # Write individual TOML file
        try:
            with open(output_file, "wb") as f:
                tomli_w.dump(entry, f)
        except Exception as e:
            print(f"Error writing {output_file}: {e}")

    if dry_run:
        print(f"\nDry run complete. Would create {total} files in {len(week_counts)} week folders")
    else:
        print(f"\nSplit complete: {total} files created in {len(week_counts)} week folders")
        for week, count in sorted(week_counts.items()):
            print(f"  {week}: {count} experiments")

    # Create backup if requested
    if backup and not dry_run:
        backup_path = metrics_path.with_suffix(".toml.bak")
        shutil.copy2(metrics_path, backup_path)
        print(f"\nBackup created: {backup_path}")

    return {
        "total": total,
        "split": total,
        "weeks": len(week_counts),
        "week_counts": week_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Split experiments/metrics.toml into week-organized files"
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("experiments/metrics.toml"),
        help="Path to metrics.toml (default: experiments/metrics.toml)",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("experiments/result"),
        help="Output directory for split files (default: experiments/result)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of original metrics.toml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )

    args = parser.parse_args()

    stats = split_metrics(
        metrics_path=args.metrics_path,
        result_dir=args.result_dir,
        backup=not args.no_backup,
        dry_run=args.dry_run,
    )

    return 0 if stats.get("errors", 0) == 0 else 1


if __name__ == "__main__":
    exit(main())

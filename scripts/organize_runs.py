"""Organize run artifacts into raw_data, results, json, images, and model folders."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DESTINATIONS = {
    "raw_data": {"telemetry.csv"},
    "results": {
        "actuals.csv",
        "predictions.csv",
        "train_losses.csv",
        "evaluation_comparison.csv",
        "evaluation_spikes.csv",
    },
    "json": {
        "metrics.json",
        "spike_summary.json",
        "scaler_params.json",
        "evaluation_summary.json",
        "model_metadata.json",
        "model_readable_summary.json",
    },
    "images": {
        "traffic_prediction_dashboard.png",
        "model_evaluation_dashboard.png",
    },
    "model": {
        "lstm_model.pth",
        "model_readable_report.md",
        "model_weights_summary.csv",
        "model_gate_summary.csv",
    },
}


def organize_run(run_dir: Path, copy: bool = False) -> list[str]:
    moved: list[str] = []
    for folder, names in DESTINATIONS.items():
        target_dir = run_dir / folder
        target_dir.mkdir(exist_ok=True)
        for name in names:
            source = run_dir / name
            target = target_dir / name
            if not source.exists() or target.exists():
                continue
            if copy:
                shutil.copy2(source, target)
                moved.append(f"copied {source} -> {target}")
            else:
                shutil.move(str(source), str(target))
                moved.append(f"moved {source} -> {target}")
    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs", help="Root folder containing run directories.")
    parser.add_argument("--run-dir", default=None, help="Only organize one run directory.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of moving them.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(args.run_dir)] if args.run_dir else [path for path in Path(args.runs_root).iterdir() if path.is_dir()]
    total = 0
    for run_dir in run_dirs:
        changes = organize_run(run_dir, args.copy)
        total += len(changes)
        if changes:
            print(f"{run_dir}:")
            for change in changes:
                print(f"  {change}")
    print(f"Organized {len(run_dirs)} run folder(s), {total} file operation(s).")


if __name__ == "__main__":
    main()

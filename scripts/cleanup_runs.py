"""Clean empty and incomplete run folders."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


COMPLETE_MARKERS = [
    "raw_data/telemetry.csv",
    "results/predictions.csv",
    "results/actuals.csv",
    "results/train_losses.csv",
    "json/metrics.json",
    "images/traffic_prediction_dashboard.png",
    "model/lstm_model.pth",
]


def is_empty_dir(path: Path) -> bool:
    return path.is_dir() and not any(path.iterdir())


def remove_empty_dirs(root: Path) -> int:
    removed = 0
    for path in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        if is_empty_dir(path):
            path.rmdir()
            removed += 1
    return removed


def is_complete_run(run_dir: Path) -> bool:
    return all((run_dir / marker).exists() for marker in COMPLETE_MARKERS)


def is_raw_only_run(run_dir: Path) -> bool:
    files = [path for path in run_dir.rglob("*") if path.is_file()]
    return len(files) == 1 and files[0].as_posix().endswith("/raw_data/telemetry.csv")


def cleanup_runs(runs_root: Path, remove_incomplete: bool = False) -> list[str]:
    actions: list[str] = []
    runs_root.mkdir(exist_ok=True)
    for run_dir in sorted([path for path in runs_root.iterdir() if path.is_dir()]):
        removed_empty = remove_empty_dirs(run_dir)
        if removed_empty:
            actions.append(f"{run_dir}: removed {removed_empty} empty subfolder(s)")

        if remove_incomplete and not is_complete_run(run_dir):
            shutil.rmtree(run_dir)
            actions.append(f"{run_dir}: removed incomplete run folder")
            continue

        if is_raw_only_run(run_dir) and run_dir.name in {"kaggle_wsl", "quick_live"}:
            shutil.rmtree(run_dir)
            actions.append(f"{run_dir}: removed staging/raw-only folder")

    remove_empty_dirs(runs_root)
    return actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs", help="Root folder containing run directories.")
    parser.add_argument(
        "--remove-incomplete",
        action="store_true",
        help="Remove any run folder that does not contain a complete model output set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = cleanup_runs(Path(args.runs_root), args.remove_incomplete)
    if actions:
        print("\n".join(actions))
    else:
        print("No cleanup needed.")


if __name__ == "__main__":
    main()

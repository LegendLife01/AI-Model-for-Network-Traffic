"""Summarize saved trial runs by quality, baseline gain, and artifact status."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    runs_dir = Path("runs")
    rows = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        summary_path = run_dir / "json" / "evaluation_summary.json"
        traffic_image = run_dir / "images" / "traffic_prediction_dashboard.png"
        eval_image = run_dir / "images" / "model_evaluation_dashboard.png"
        row = {
            "run": run_dir.name,
            "quality": None,
            "gain": None,
            "rows": None,
            "epochs": None,
            "model": "unknown",
            "traffic_graph": traffic_image.exists(),
            "eval_graph": eval_image.exists(),
        }
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            overall = summary.get("overall", {})
            benchmark = summary.get("benchmark", {})
            row.update(
                {
                    "quality": overall.get("normalized_quality_pct"),
                    "gain": overall.get("mae_improvement_vs_persistence_pct"),
                    "rows": overall.get("rows"),
                    "epochs": overall.get("epochs"),
                    "model": benchmark.get("artifact", "unknown"),
                }
            )
        rows.append(row)

    rows.sort(key=lambda item: (-1 if item["quality"] is None else item["quality"]), reverse=True)
    print(f"{'Run':<40} {'Quality':>8} {'Gain':>9} {'Rows':>6} {'Epochs':>6} {'Graphs':>8}  Model")
    print("-" * 100)
    for row in rows:
        quality = "n/a" if row["quality"] is None else f"{row['quality']:.2f}"
        gain = "n/a" if row["gain"] is None else f"{row['gain']:.2f}"
        rows_value = "n/a" if row["rows"] is None else str(row["rows"])
        epochs = "n/a" if row["epochs"] is None else str(row["epochs"])
        graphs = f"{'T' if row['traffic_graph'] else '-'}{'E' if row['eval_graph'] else '-'}"
        print(f"{row['run']:<40} {quality:>8} {gain:>9} {rows_value:>6} {epochs:>6} {graphs:>8}  {row['model']}")


if __name__ == "__main__":
    main()

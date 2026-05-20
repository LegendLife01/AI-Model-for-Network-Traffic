"""Load Kaggle computer-network-traffic data into telemetry.csv format.

The Crawford dataset has daily network flow counts with columns:
date, l_ipn, r_asn, and f. This script converts those flows into the project
schema used by the LSTM:
timestamp, traffic_mbps, latency_ms, packet_loss_pct, source.

Each run randomizes the selected workstation profile, scaling, and jitter unless
--seed is provided. That keeps repeated training runs from seeing identical
network numbers while still grounding the series in the Kaggle data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = "crawford/computer-network-traffic"
DEFAULT_FILE = "cs448b_ipasn.csv"
FEATURES = ["traffic_mbps", "latency_ms", "packet_loss_pct"]


def load_kaggle_frame(file_path: str) -> pd.DataFrame:
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
    except ImportError as exc:
        raise SystemExit(
            "Missing kagglehub. Install it with: python -m pip install 'kagglehub[pandas-datasets]'"
        ) from exc

    if file_path:
        return kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            DATASET,
            file_path,
        )

    dataset_dir = Path(kagglehub.dataset_download(DATASET))
    csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset: {dataset_dir}")
    return pd.read_csv(csv_files[0])


def build_telemetry(df: pd.DataFrame, rows: int, seed: int | None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    required = {"date", "l_ipn", "f"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Kaggle dataframe is missing columns: {', '.join(sorted(missing))}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["f"] = pd.to_numeric(data["f"], errors="coerce")
    data = data.dropna(subset=["date", "l_ipn", "f"])

    workstations = sorted(data["l_ipn"].unique().tolist())
    selected = rng.choice(workstations)
    profile = (
        data[data["l_ipn"] == selected]
        .groupby("date", as_index=False)["f"]
        .sum()
        .sort_values("date")
    )

    if len(profile) < 2:
        raise ValueError("Selected workstation profile has too few rows.")

    base_flows = profile["f"].to_numpy(dtype=float)
    base_dates = profile["date"].to_numpy()
    repeats = int(np.ceil(rows / len(base_flows)))
    flows = np.tile(base_flows, repeats)[:rows]
    dates = pd.date_range(pd.Timestamp(base_dates[0]), periods=rows, freq="h")

    scale = rng.uniform(0.75, 1.45)
    jitter = rng.normal(1.0, 0.08, rows)
    randomized_flows = np.clip(flows * scale * jitter, 0, None)

    flow_min = float(np.min(randomized_flows))
    flow_range = float(np.max(randomized_flows) - flow_min)
    normalized = (randomized_flows - flow_min) / flow_range if flow_range else np.zeros(rows)

    traffic_floor = rng.uniform(8.0, 18.0)
    traffic_span = rng.uniform(90.0, 180.0)
    traffic = traffic_floor + traffic_span * normalized

    burst_count = max(3, rows // 40)
    burst_idx = rng.choice(rows, size=min(burst_count, rows), replace=False)
    traffic[burst_idx] += rng.uniform(25.0, 95.0, len(burst_idx))

    latency = 2.0 + normalized * rng.uniform(8.0, 22.0) + rng.normal(0, 0.35, rows)
    latency[burst_idx] += rng.uniform(4.0, 16.0, len(burst_idx))
    latency = np.clip(latency, 0.5, None)

    packet_loss = np.clip(rng.normal(0.04, 0.03, rows), 0, None)
    high_load = traffic > np.quantile(traffic, 0.85)
    packet_loss[high_load] += rng.uniform(0.2, 2.5, high_load.sum())
    packet_loss[burst_idx] += rng.uniform(0.8, 5.5, len(burst_idx))
    packet_loss = np.clip(packet_loss, 0.0, 100.0)

    telemetry = pd.DataFrame(
        {
            "timestamp": dates,
            "traffic_mbps": np.round(traffic, 3),
            "latency_ms": np.round(latency, 3),
            "packet_loss_pct": np.round(packet_loss, 3),
            "source": f"kaggle:{DATASET}:l_ipn={selected}",
        }
    )
    return telemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="telemetry.csv", help="CSV path to write.")
    parser.add_argument("--rows", type=int, default=720, help="Telemetry rows to generate.")
    parser.add_argument("--file-path", default=DEFAULT_FILE, help="Kaggle file path to load.")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_kaggle_frame(args.file_path)
    telemetry = build_telemetry(raw, args.rows, args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(output, index=False)
    print(f"Loaded Kaggle dataset {DATASET}")
    print(f"Generated {len(telemetry)} telemetry rows -> {output}")
    print(telemetry[FEATURES].describe().round(3).to_string())


if __name__ == "__main__":
    main()

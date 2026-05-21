"""Load Kaggle computer-network-traffic data into telemetry.csv format.

The Crawford dataset has network flow counts with columns:
date, l_ipn, r_asn, and f. This script converts those flow counts into the
project schema used by the model:
timestamp, traffic_mbps, latency_ms, packet_loss_pct, source.

The conversion always keeps rows sorted by timestamp so model training remains
a valid time-series task. If --seed is omitted, each run can use a different
chronological slice. It does not repeat old rows to pad the dataset; if a
requested sample count is larger than the available real slice, it uses the
available rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = "crawford/computer-network-traffic"
DEFAULT_FILE = "cs448b_ipasn.csv"
FEATURES = ["traffic_mbps", "latency_ms", "packet_loss_pct"]


def cached_dataset_file(file_path: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets" / "crawford" / "computer-network-traffic"
    if not cache_root.exists():
        return None
    target_name = Path(file_path or DEFAULT_FILE).name
    matches = sorted(cache_root.rglob(target_name), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def load_kaggle_frame(file_path: str) -> pd.DataFrame:
    cached = cached_dataset_file(file_path)
    if cached is not None:
        print(f"Using cached Kaggle CSV: {cached}")
        return pd.read_csv(cached)

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Missing kagglehub. Install it with: python -m pip install 'kagglehub[pandas-datasets]'"
        ) from exc

    dataset_dir = Path(kagglehub.dataset_download(DATASET))
    csv_files = sorted(dataset_dir.rglob(Path(file_path or DEFAULT_FILE).name))
    if not csv_files:
        csv_files = sorted(dataset_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset: {dataset_dir}")
    return pd.read_csv(csv_files[0])


def build_telemetry(
    df: pd.DataFrame,
    rows: int,
    seed: int | None,
    l_ipn: int | None,
    augment: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    required = {"date", "l_ipn", "f"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Kaggle dataframe is missing columns: {', '.join(sorted(missing))}")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["f"] = pd.to_numeric(data["f"], errors="coerce")
    data = data.dropna(subset=["date", "l_ipn", "f"])

    data["l_ipn"] = pd.to_numeric(data["l_ipn"], errors="coerce")
    data = data.dropna(subset=["l_ipn"]).copy()
    data["l_ipn"] = data["l_ipn"].astype(int)

    workstations = sorted(data["l_ipn"].unique().tolist())
    if l_ipn is not None:
        if l_ipn not in workstations:
            raise ValueError(f"l_ipn={l_ipn} not found. Available examples: {workstations[:10]}")
        group_cols = ["date", "l_ipn"] + (["r_asn"] if "r_asn" in data.columns else [])
        profile = (
            data[data["l_ipn"] == l_ipn]
            .groupby(group_cols, as_index=False)["f"]
            .sum()
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
        source_label = f"l_ipn={l_ipn}"
    else:
        l_ipn = int(data["l_ipn"].value_counts().idxmax())
        group_cols = ["date", "l_ipn"] + (["r_asn"] if "r_asn" in data.columns else [])
        profile = (
            data[data["l_ipn"] == l_ipn]
            .groupby(group_cols, as_index=False)["f"]
            .sum()
            .sort_values(group_cols)
            .reset_index(drop=True)
        )
        source_label = f"auto_most_active_l_ipn={l_ipn}"

    if len(profile) < 2:
        raise ValueError("Selected workstation profile has too few rows.")

    requested_rows = max(1, rows)
    if requested_rows < len(profile):
        max_start = len(profile) - requested_rows
        start = int(rng.integers(0, max_start + 1)) if seed is None else int(rng.integers(0, max_start + 1))
        profile = profile.iloc[start : start + requested_rows].copy()
    else:
        profile = profile.copy()

    flow_series = profile["f"].to_numpy(dtype=float)
    if augment:
        scale = rng.uniform(0.75, 1.45)
        jitter = rng.normal(1.0, 0.08, len(flow_series))
        flow_series = flow_series * scale * jitter
    flow_series = np.clip(flow_series, 0, None)

    row_offsets = profile.groupby("date").cumcount()
    dates = pd.to_datetime(profile["date"]) + pd.to_timedelta(row_offsets, unit="m")
    rows = len(flow_series)

    flow_min = float(np.min(flow_series))
    flow_range = float(np.max(flow_series) - flow_min)
    normalized = (flow_series - flow_min) / flow_range if flow_range else np.zeros(rows)

    traffic_floor = 10.0
    traffic_span = 140.0
    traffic = traffic_floor + traffic_span * normalized

    if augment:
        burst_count = max(3, rows // 40)
        burst_idx = rng.choice(rows, size=min(burst_count, rows), replace=False)
        traffic[burst_idx] += rng.uniform(25.0, 95.0, len(burst_idx))
    else:
        burst_idx = np.array([], dtype=int)

    latency = 2.0 + normalized * 14.0
    if augment:
        latency += rng.normal(0, 0.35, rows)
        latency[burst_idx] += rng.uniform(4.0, 16.0, len(burst_idx))
    latency = np.clip(latency, 0.5, None)

    packet_loss = np.zeros(rows)
    high_load = traffic > np.quantile(traffic, 0.88)
    burst_mask = np.zeros(rows, dtype=bool)
    burst_mask[burst_idx] = True
    coupled_spikes = high_load | burst_mask
    packet_loss[coupled_spikes] += 0.2 + 2.3 * normalized[coupled_spikes]
    if augment:
        packet_loss += np.clip(rng.normal(0.015, 0.015, rows), 0, None)
        packet_loss[burst_idx] += rng.uniform(0.8, 5.5, len(burst_idx))
    packet_loss = np.clip(packet_loss, 0.0, 100.0)

    telemetry = pd.DataFrame(
        {
            "timestamp": dates,
            "traffic_mbps": np.round(traffic, 3),
            "latency_ms": np.round(latency, 3),
            "packet_loss_pct": np.round(packet_loss, 3),
            "source": f"kaggle-flow-converted:{DATASET}:{source_label}",
        }
    )
    telemetry["traffic_delta"] = telemetry["traffic_mbps"].diff().fillna(0.0).round(3)
    telemetry["latency_delta"] = telemetry["latency_ms"].diff().fillna(0.0).round(3)
    return telemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="telemetry.csv", help="CSV path to write.")
    parser.add_argument("--rows", type=int, default=8000, help="Telemetry rows to generate.")
    parser.add_argument("--file-path", default=DEFAULT_FILE, help="Kaggle file path to load.")
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducible seed.")
    parser.add_argument("--l-ipn", type=int, default=None, help="Optional local workstation ID to use. Defaults to the most active workstation.")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Add random scaling, jitter, and coupled bursts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_kaggle_frame(args.file_path)
    telemetry = build_telemetry(raw, args.rows, args.seed, args.l_ipn, args.augment)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(output, index=False)
    thresholds = {
        feature: float(np.quantile(telemetry[feature].to_numpy(dtype=float), 0.90))
        for feature in FEATURES
    }
    sidecar = output.with_suffix(".spike_thresholds.json")
    sidecar.write_text(json.dumps({"spike_thresholds": thresholds, "quantile": 0.90}, indent=2), encoding="utf-8")
    print(f"Loaded Kaggle dataset {DATASET}")
    print(f"Generated {len(telemetry)} telemetry rows -> {output}")
    print(f"Saved spike threshold sidecar -> {sidecar}")
    print(telemetry[FEATURES].describe().round(3).to_string())


if __name__ == "__main__":
    main()

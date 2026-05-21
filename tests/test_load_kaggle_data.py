import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.load_kaggle_data import build_telemetry


def test_kaggle_loader_preserves_chronological_order_and_columns():
    raw = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="h").repeat(2),
            "l_ipn": [1, 2] * 8,
            "r_asn": [10, 20] * 8,
            "f": [10, 2, 12, 3, 14, 4, 60, 5, 16, 6, 18, 7, 70, 8, 20, 9],
        }
    )

    telemetry = build_telemetry(raw, rows=6, seed=42, l_ipn=None, augment=True)

    assert list(telemetry.columns) == [
        "timestamp",
        "traffic_mbps",
        "latency_ms",
        "packet_loss_pct",
        "source",
        "traffic_delta",
        "latency_delta",
    ]
    assert telemetry["timestamp"].is_monotonic_increasing
    assert telemetry["source"].str.contains("auto_most_active_l_ipn").all()
    assert (telemetry[["traffic_mbps", "latency_ms", "packet_loss_pct"]] >= 0).all().all()


if __name__ == "__main__":
    test_kaggle_loader_preserves_chronological_order_and_columns()

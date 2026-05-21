import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.metrics_utils import compute_spike_scores, quality_score_v2, spike_thresholds_from_quantile, summarize_gates, weighted_mae
from ml.telemetry_profile import profile_telemetry


def test_spike_thresholds_and_scores_detect_predicted_spikes():
    train = np.array([[1.0, 10.0, 0.0], [2.0, 11.0, 0.1], [10.0, 30.0, 2.0]])
    thresholds = spike_thresholds_from_quantile(train, 0.8)
    actuals = np.array([[1.5, 10.5, 0.0], [12.0, 35.0, 2.5]])
    predictions = np.array([[1.4, 10.3, 0.0], [11.0, 34.0, 2.2]])

    scores = compute_spike_scores(actuals, predictions, thresholds)

    assert scores["traffic_mbps"]["f1"] == 1.0
    assert scores["latency_ms"]["predicted_spikes"] == 1
    assert scores["packet_loss_pct"]["actual_spikes"] == 1


def test_quality_score_weights_features():
    per_feature = {
        "traffic_mbps": {"model_mae": 1.0, "baseline_mae": 4.0, "r2": 0.8, "spike_f1": 0.7, "actual_spikes": 5},
        "latency_ms": {"model_mae": 1.0, "baseline_mae": 2.0, "r2": 0.5, "spike_f1": 0.5, "actual_spikes": 5},
        "packet_loss_pct": {"model_mae": 0.1, "baseline_mae": 0.2, "r2": 0.4, "spike_f1": 0.4, "actual_spikes": 5},
    }

    assert 0.0 < quality_score_v2(per_feature) <= 100.0
    assert weighted_mae(np.zeros((2, 3)), np.ones((2, 3))) == 1.0


def test_gates_all_true_when_quality_high():
    rows = [
        {"metric": "traffic_mbps", "mae_improvement_pct": 20.0},
        {"metric": "latency_ms", "mae_improvement_pct": 10.0},
        {"metric": "packet_loss_pct", "mae_improvement_pct": 5.0},
    ]
    gates = summarize_gates(91.0, 15.0, rows, 0.6, 5, 92.0, 60.0)
    assert all(gates.values())


def test_profile_recommends_gb_on_small_rows(tmp_path):
    rows = ["timestamp,traffic_mbps,latency_ms,packet_loss_pct"]
    for idx in range(150):
        rows.append(f"2026-01-01 00:{idx % 60:02d}:00,{10 + idx % 3},{2 + (idx % 2) * 0.1},0.01")
    data = tmp_path / "small.csv"
    data.write_text("\n".join(rows), encoding="utf-8")
    profile = profile_telemetry(data)
    assert profile.recommended_trainer == "gb_only"


if __name__ == "__main__":
    test_spike_thresholds_and_scores_detect_predicted_spikes()
    test_quality_score_weights_features()
    import tempfile
    from pathlib import Path

    test_gates_all_true_when_quality_high()
    with tempfile.TemporaryDirectory() as tmp:
        test_profile_recommends_gb_on_small_rows(Path(tmp))

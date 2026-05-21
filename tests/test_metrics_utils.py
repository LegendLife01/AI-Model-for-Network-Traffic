import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.metrics_utils import compute_spike_scores, quality_score_v2, spike_thresholds_from_quantile, weighted_mae


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


if __name__ == "__main__":
    test_spike_thresholds_and_scores_detect_predicted_spikes()
    test_quality_score_weights_features()

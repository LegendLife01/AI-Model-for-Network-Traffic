"""Shared scoring helpers for telemetry forecasting."""

from __future__ import annotations

import numpy as np


FEATURES = ["traffic_mbps", "latency_ms", "packet_loss_pct"]
FEATURE_WEIGHTS = {"traffic_mbps": 0.50, "latency_ms": 0.25, "packet_loss_pct": 0.25}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def spike_thresholds_from_train(y_train: np.ndarray, multiplier: float = 1.0) -> dict[str, float]:
    return {
        feature: float(y_train[:, idx].mean() + multiplier * y_train[:, idx].std(ddof=0))
        for idx, feature in enumerate(FEATURES)
    }


def spike_thresholds_from_quantile(y_train: np.ndarray, quantile: float = 0.90) -> dict[str, float]:
    return {
        feature: float(np.quantile(y_train[:, idx], quantile))
        for idx, feature in enumerate(FEATURES)
    }


def compute_spike_scores(actuals: np.ndarray, predictions: np.ndarray, thresholds: dict[str, float]) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for idx, feature in enumerate(FEATURES):
        threshold = float(thresholds[feature])
        actual_spikes = actuals[:, idx] > threshold
        predicted_spikes = predictions[:, idx] > threshold
        true_positive = int(np.logical_and(actual_spikes, predicted_spikes).sum())
        false_positive = int(np.logical_and(~actual_spikes, predicted_spikes).sum())
        false_negative = int(np.logical_and(actual_spikes, ~predicted_spikes).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
        scores[feature] = {
            "threshold": threshold,
            "actual_spikes": int(actual_spikes.sum()),
            "predicted_spikes": int(predicted_spikes.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return scores


def weighted_mae(actuals: np.ndarray, predictions: np.ndarray, weights: dict[str, float] = FEATURE_WEIGHTS) -> float:
    values = []
    for idx, feature in enumerate(FEATURES):
        values.append(weights[feature] * float(np.mean(np.abs(actuals[:, idx] - predictions[:, idx]))))
    return float(sum(values))


def feature_quality(model_mae: float, baseline_mae: float, r2: float, spike_f1: float, actual_spikes: int) -> float:
    mae_ratio = model_mae / max(baseline_mae, 1e-9)
    error_score = clamp(1.0 - mae_ratio) * 0.55 + clamp(r2) * 0.45
    spike_score = 0.75 if actual_spikes < 3 else clamp(spike_f1)
    return 100.0 * (0.65 * error_score + 0.35 * spike_score)


def quality_score_v2(per_feature: dict[str, dict[str, float]], weights: dict[str, float] = FEATURE_WEIGHTS) -> float:
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0
    score = 0.0
    for feature in FEATURES:
        row = per_feature[feature]
        score += weights[feature] * feature_quality(
            float(row["model_mae"]),
            float(row["baseline_mae"]),
            float(row.get("r2", 0.0)),
            float(row.get("spike_f1", 0.0)),
            int(row.get("actual_spikes", 0)),
        )
    return float(score / total_weight)

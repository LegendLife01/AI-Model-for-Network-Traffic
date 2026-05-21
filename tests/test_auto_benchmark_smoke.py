import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.auto_benchmark import run_benchmark


def test_auto_benchmark_smoke_creates_log(tmp_path):
    rows = ["timestamp,traffic_mbps,latency_ms,packet_loss_pct"]
    for idx in range(160):
        traffic = 20 + (idx % 24) * 2
        if idx % 41 == 0:
            traffic += 60
        latency = 2 + (traffic / 100)
        loss = 0.01 + (0.8 if traffic > 70 else 0.0)
        rows.append(f"2026-01-01 00:{idx % 60:02d}:00,{traffic:.3f},{latency:.3f},{loss:.3f}")
    data = tmp_path / "telemetry.csv"
    data.write_text("\n".join(rows), encoding="utf-8")
    out = tmp_path / "bench"
    result = run_benchmark(data, out, target_quality=85, max_attempts=1, max_minutes=5)
    assert result.attempts == 1
    assert (out / "benchmark_log.jsonl").exists()
    assert (out / "best_run.txt").exists()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_auto_benchmark_smoke_creates_log(Path(tmp))

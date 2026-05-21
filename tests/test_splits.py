import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ml"))

from ml.enhanced_train import gb_slice_for_sequences, original_index_for_sequence
from ml.train_model import create_sequences


def test_create_sequences_keeps_next_step_target_order():
    data = np.arange(10, dtype=np.float32).reshape(-1, 1)
    x, y = create_sequences(data, 3)

    assert x[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert y[0, 0] == 3.0
    assert x[-1, :, 0].tolist() == [6.0, 7.0, 8.0]
    assert y[-1, 0] == 9.0


def test_gb_sequence_slice_does_not_point_after_sequence_targets():
    seq_len = 48
    lookback = 24
    split_start = 100
    split_end = 150
    gb_slice = gb_slice_for_sequences(split_start, split_end, seq_len, lookback, total_rows=1000)

    assert gb_slice.start == original_index_for_sequence(split_start, seq_len) - lookback
    assert gb_slice.stop == original_index_for_sequence(split_end, seq_len) - lookback
    assert gb_slice.start < gb_slice.stop


if __name__ == "__main__":
    test_create_sequences_keeps_next_step_target_order()
    test_gb_sequence_slice_does_not_point_after_sequence_targets()

# Codex Spec: Universal Telemetry Benchmark (≥90% Quality Loop)

**Repo:** `AI-Model-for-Network-Traffic-main` (latest zip)  
**Goal:** For **any** input CSV matching schema `timestamp,traffic_mbps,latency_ms,packet_loss_pct`, automatically train + calibrate until `evaluation_summary.json → gates_passed` are **all true** and `overall.normalized_quality_pct >= 90`.

**Current measured state (do not regress):**
| Dataset | Quality | MAE vs persistence | Traffic spike F1 | Gates |
|---------|---------|-------------------|------------------|-------|
| Synthetic 2000 rows | **78.3%** | +24.7% | **0.71** | `quality_ge_90` ❌ |
| Kaggle ~4000 rows | **57.2%** | +3.2% | **0.15** | traffic MAE ❌ persistence |

Codex must implement the **auto-benchmark loop** below and iterate until both reference sets pass, then generalize to arbitrary `ml/telemetry.csv` or `--data` path.

---

## 0. Codex master prompt (paste this first)

```
You are working in the AI-Model-for-Network-Traffic-main repository root.

READ: CODEX_ITERATIVE_BENCHMARK.md (this file) end-to-end.

MISSION: Implement Phases A→H so that ANY valid telemetry CSV can reach ≥90% normalized_quality_pct
using an automated retry loop (not manual hyperparameter guessing).

WORK RULES:
1. Implement phases in order; run tests/smoke after each phase.
2. Do not break existing CLI flags; add new commands alongside.
3. Never shuffle time-series rows; never fit scalers on val/test.
4. Do not fake metrics—improve predictions and spike recall, not just formulas.
5. After each phase, run the verification commands in Section 8.
6. LOOP until Section 8 passes on: (a) synthetic 2000 rows, (b) Kaggle 5000+ rows, (c) ml/telemetry.csv if present.
7. Print final metrics table and copy best artifacts to docs/results/ and docs/images/.

START with Phase A (telemetry_profile.py + auto_benchmark.py skeleton), then B→H.
If stuck after 12 attempts on a dataset, print diagnosis from profile.json and stop with actionable errors.

When finished, output:
- overall quality for synthetic + kaggle + generic telemetry
- gates_passed JSON for each
- path to best run folder
```

---

## 1. Benchmark definition (single source of truth)

**File:** `ml/metrics_utils.py` (extend, do not duplicate)

### 1.1 Gates (all must be `true` in `gates_passed`)

| Gate key | Rule |
|----------|------|
| `quality_ge_90` | `overall.normalized_quality_pct >= 90.0` |
| `mae_improvement_ge_15` | mean per-feature MAE improvement vs persistence ≥ 15% |
| `beats_persistence_each_feature_mae` | each feature `mae_improvement_pct > 0` |
| `traffic_spike_f1_ge_0_50` | traffic spike F1 ≥ 0.50 |
| `traffic_predicted_spikes_ge_5` | traffic predicted_spikes ≥ 5 |
| `model_quality_gt_persistence` | weighted per-feature quality > persistence (use same formula as overall) |

### 1.2 Overall quality formula (already in repo — keep)

```python
overall_quality = weighted_mean(feature_quality(...), FEATURE_WEIGHTS)
# traffic 0.50, latency 0.25, packet_loss 0.25
```

### 1.3 Fix baseline table inconsistency (Phase B)

**Bug today:** `summarize_baselines()` in `evaluate_model.py` averages per-feature qualities into a table that can disagree with `overall` (e.g. Model 38% vs overall 78%).

**Fix:** Set each method's `quality_pct` = same weighted overall as `quality_score_v2` built from per-feature rows. Remove alternate averaging paths.

---

## 2. Phase A — Telemetry profiler (any CSV)

**CREATE:** `ml/telemetry_profile.py`

```python
@dataclass
class TelemetryProfile:
    rows: int
    usable_rows: int
    train_rows_est: int
    traffic_std: float
    latency_std: float
    loss_std: float
    traffic_spike_rate: float  # fraction > q90 on train slice
    persistence_mae: dict[str, float]  # per feature on val slice
    volatility: str  # low | medium | high
    recommended_sequence_length: int
    recommended_lookback: int
    recommended_spike_quantile: float
    recommended_trainer: str  # hybrid | gb_only | hybrid_aggressive
    recommended_epochs: int

def profile_telemetry(path: Path, train_ratio=0.70, test_ratio=0.82) -> TelemetryProfile:
    """
    - Auto sequence_length: clamp(int(usable_rows * 0.05), 24, 96)
    - Auto lookback: min(24, sequence_length // 2)
    - If traffic_spike_rate < 0.05: spike_quantile = 0.85 else 0.90
    - If persistence already strong (weighted MAE < 0.5 on val): recommend gb_only
    - If rows < 400: recommend gb_only (LSTM unstable)
    - If traffic coefficient of variation > 0.6: hybrid_aggressive
    """
```

**CLI:** `python ml/telemetry_profile.py --data path/to.csv --output json/profile.json`

---

## 3. Phase B — Validation calibration layer (any telemetry)

**CREATE:** `ml/calibrate_predictions.py`

Purpose: After base model predicts on **validation**, learn a **non-leaky** adjustment that lifts spike F1 without destroying MAE. Apply same adjustment to test predictions.

```python
def calibrate(
    val_actuals: np.ndarray,
    val_predictions: np.ndarray,
    thresholds: dict[str, float],
) -> CalibrationParams:
    """
    Per feature j, search small grid on validation only:
      pred' = a[j] * pred + b[j] + c[j] * persistence_val
    with constraints:
      - a in [0.85, 1.15], b in [-0.1*std, 0.1*std], c in [0, 0.5]
    Objective = feature_quality proxy (same as metrics_utils)
    + penalty if predicted_spikes < 5 for traffic

    Optional spike boost for traffic:
      if pred[j] > threshold * 0.92 and actual spike history suggests under-shoot:
        pred'[j] = max(pred'[j], threshold * 1.02)
    """
```

**Integrate into:** `enhanced_train.py` **before** writing `predictions.csv`:
1. Split: use val slice predictions + actuals.
2. `params = calibrate(val_actuals, val_pred, thresholds)`
3. `final_pred = apply_calibration(test_pred, params, test_persistence)`

Store `calibration` dict in `metrics.json` under `training`.

---

## 4. Phase C — Trainer tournament (pick best for this telemetry)

**CREATE:** `ml/trainer_tournament.py`

Given `TelemetryProfile`, run **candidates** on the same chronological split (fast mode: fewer epochs for screening, full epochs for winner).

| Candidate ID | Trainer | Notes |
|--------------|---------|-------|
| `hybrid_default` | `enhanced_train.py` | profile seq_len, quantile, lr |
| `hybrid_aggressive` | enhanced | spike_weight=8, focal_gamma=1.0, gb_depth=5 |
| `gb_spike` | `train_kaggle_model.py` | spike_oversample=3, lookback=profile |
| `hybrid_short_seq` | enhanced | seq_len=48 if rows medium |
| `hybrid_gb_heavy` | enhanced | force initial gb_weight=0.85 per traffic |

**Screening score** (validation only, no test peeking):

```python
screen_score = 0.55 * weighted_mae_gain_vs_persistence + 0.45 * weighted_spike_f1
```

Pick top 1–2 candidates for full training.

---

## 5. Phase D — Auto-benchmark loop (core deliverable)

**CREATE:** `ml/auto_benchmark.py`

```python
def run_benchmark(
    data: Path,
    output_dir: Path,
    target_quality: float = 90.0,
    max_attempts: int = 12,
    max_minutes: int = 45,
) -> BenchmarkResult:
```

### Algorithm (implement exactly)

```
profile = profile_telemetry(data)
attempts = []
for attempt in range(max_attempts):
    candidate = next_candidate(profile, attempts)  # tournament order + adaptive
    run_dir = output_dir / f"attempt_{attempt:02d}_{candidate.id}"
    train_candidate(candidate, data, run_dir)
    evaluate_model(run_dir)
    summary = load evaluation_summary.json
    attempts.append({candidate, summary, run_dir})

    if all(summary.gates_passed.values()) and summary.overall.normalized_quality_pct >= target_quality:
        mark BEST, sync docs, return SUCCESS

    # Adaptive mutations for next attempt based on failed gates:
    if not beats_persistence_each_feature_mae:
        if traffic mae_improvement < 0: next = gb_spike or hybrid_gb_heavy
        else: next = increase persistence blend search on val
    if not traffic_spike_f1_ge_0_50:
        next = lower spike_quantile (0.85), raise spike_weight, enable calibrate spike boost
    if quality < 90 but close (>= 82):
        next = enable full calibration grid + more epochs (+20)
    if quality < 75:
        next = gb_only with spike_oversample=4

print failure report with best attempt
return BEST_EFFORT
```

### CLI

```bash
python ml/auto_benchmark.py --data ml/telemetry.csv --output-dir runs/auto_telemetry --target-quality 90 --max-attempts 12
python ml/auto_benchmark.py --data <ANY.csv> --target-quality 90
```

### Persist

- `runs/auto_*/benchmark_log.jsonl` — one JSON line per attempt
- `runs/auto_*/best_run.txt` — path to winning run
- Copy winner to `docs/results/generic_*` when `--sync-docs`

---

## 6. Phase E — Wire runners

### `run.ps1` / `run.sh` — add mode `benchmark`

```powershell
param(
  ...
  [ValidateSet(..., "benchmark")]
  ...
  [double]$TargetQuality = 90
  [int]$MaxAttempts = 12
)

# benchmark mode:
#   if Mode=kaggle: load kaggle -> raw_data/telemetry.csv
#   if Mode=synthetic: generate_data
#   if Mode=train: use ml/telemetry.csv
#   python ml/auto_benchmark.py --data $DataFile --output-dir $RunDir --target-quality $TargetQuality --max-attempts $MaxAttempts
#   python ml/evaluate_model.py --run-dir (best from benchmark)
#   python ml/visualize.py ...
#   python scripts/analyze_runs.py --sync-docs --run <best> --prefix generic_
```

### `run.ps1 train` change

Default `train` should call `auto_benchmark.py` instead of raw `enhanced_train.py` when env `AUTO_BENCHMARK=1` or new flag `-AutoBenchmark` (default **true**).

---

## 7. Phase F — Model fixes still needed (latest zip gaps)

Implement inside `enhanced_train.py` / `train_kaggle_model.py`:

### F1. Per-target target scaling (helps packet_loss + latency)

Before sequence creation:

```python
# Store raw; train targets as z-score per feature on train slice only
# Inverse transform at prediction time
```

Already partially via StandardScaler — **verify** inverse on each head matches.

### F2. Delta + level dual prediction (traffic)

```python
# LSTM predicts delta_traffic; final traffic = last_value_in_window + delta
# Reduces persistence gap on bursty series
```

Add column in window: `traffic_last` as input feature.

### F3. Stronger GB when LSTM loses on validation

```python
if mae_lstm_val > mae_persistence_val for traffic:
    gb_weight[0] = max(gb_weight[0], 0.80)
```

### F4. Spike threshold alignment

Trainer saves `spike_thresholds` in `metrics.json` (already).  
Evaluator **must** use same quantile from training (`spike_quantile` in metrics), not recompute differently.

**Fix in `evaluate_model.py`:** read `training.spike_quantile` and `training.spike_thresholds` first.

### F5. Kaggle loader defaults (`load_kaggle_data.py`)

```python
# Default --augment True
# Default --rows 8000
# Auto-pick busiest l_ipn when omitted
```

### F6. Quality shortfall diagnosis

When `quality < 90`, write `json/benchmark_diagnosis.json`:

```json
{
  "bottleneck_feature": "packet_loss_pct",
  "bottleneck_reason": "low_r2_and_spike_f1",
  "suggested_candidate": "gb_spike"
}
```

---

## 8. Phase G — Tests & CI

**Extend** `tests/test_metrics_utils.py`:
- `test_gates_all_true_when_quality_high`
- `test_profile_recommends_gb_on_small_rows`

**CREATE** `tests/test_auto_benchmark_smoke.py`:
- 300-row synthetic CSV, `max_attempts=2`, assert benchmark_log created

**CREATE** `.github/workflows/ci.yml`:
- pytest
- `python ml/generate_data.py --hours 300 --output /tmp/t.csv`
- `python ml/auto_benchmark.py --data /tmp/t.csv --max-attempts 2 --target-quality 90` (allow fail in CI only if time; else use target 85 for CI)

---

## 9. Phase H — Documentation

Update `README.md`:
- Section **Universal benchmark mode**: `.\run.ps1 benchmark -Mode train` or any CSV
- Explain loop: profile → tournament → train → calibrate → evaluate → retry
- Update evidence tables from real `docs/results/synthetic_*` and `kaggle_*` after pass
- Add `docs/images/*.png` from best runs (no broken links)

---

## 10. Verification commands (Codex must run until green)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
python -m pip install -r requirements.txt
pytest -q

# Reference 1 — Synthetic
python ml/generate_data.py --hours 2000 --output runs/verify_synthetic/raw_data/telemetry.csv
python ml/auto_benchmark.py --data runs/verify_synthetic/raw_data/telemetry.csv --output-dir runs/verify_synthetic --target-quality 90 --max-attempts 12
python ml/evaluate_model.py --run-dir (Get-Content runs/verify_synthetic/best_run.txt)

# Reference 2 — Kaggle
python ml/load_kaggle_data.py --output runs/verify_kaggle/raw_data/telemetry.csv --rows 8000 --augment --seed 42
python ml/auto_benchmark.py --data runs/verify_kaggle/raw_data/telemetry.csv --output-dir runs/verify_kaggle --target-quality 90 --max-attempts 12

# Reference 3 — Any user telemetry
python ml/auto_benchmark.py --data ml/telemetry.csv --output-dir runs/verify_generic --target-quality 90 --max-attempts 12

# Sync docs
python scripts/analyze_runs.py --sync-docs --run (Get-Content runs/verify_kaggle/best_run.txt) --prefix kaggle_
python scripts/analyze_runs.py --sync-docs --run (Get-Content runs/verify_synthetic/best_run.txt) --prefix synthetic_
```

**Success print template:**

```
DATASET          QUALITY   GATES   TRAFFIC_F1   MAE_GAIN
synthetic        92.1      PASS    0.74         28.3%
kaggle           90.4      PASS    0.52         18.7%
generic          91.0      PASS    0.61         22.1%
```

---

## 11. Candidate mutation table (for `next_candidate`)

| Failed gate | Next candidate config |
|-------------|----------------------|
| `quality_ge_90` (78–89) | `hybrid_aggressive` + calibration + epochs+20 |
| `quality_ge_90` (<75) | `gb_spike` oversample=4 |
| `beats_persistence_each_feature_mae` (traffic) | `hybrid_gb_heavy`, lookback=36 |
| `beats_persistence_each_feature_mae` (loss) | delta traffic head + gb_only for loss column |
| `traffic_spike_f1_ge_0_50` | quantile=0.85, spike_weight=10, calibrate boost |
| `mae_improvement_ge_15` | increase persistence blend on val, hybrid_short_seq |
| All pass but quality 89.5 | calibration fine-grid only (cheap attempt) |

---

## 12. Files to create or modify (checklist)

| File | Action |
|------|--------|
| `ml/telemetry_profile.py` | **CREATE** |
| `ml/calibrate_predictions.py` | **CREATE** |
| `ml/trainer_tournament.py` | **CREATE** |
| `ml/auto_benchmark.py` | **CREATE** |
| `ml/metrics_utils.py` | extend `diagnose_quality_shortfall()` |
| `ml/enhanced_train.py` | calibration hook, delta traffic, F3 |
| `ml/evaluate_model.py` | fix baseline quality, threshold alignment |
| `ml/load_kaggle_data.py` | defaults augment/rows/l_ipn |
| `run.ps1`, `run.sh` | `benchmark` mode, `-AutoBenchmark` on train |
| `scripts/analyze_runs.py` | read `best_run.txt`, multi-prefix sync |
| `tests/test_auto_benchmark_smoke.py` | **CREATE** |
| `README.md` | universal benchmark docs |
| `docs/results/*`, `docs/images/*` | regenerate after pass |

---

## 13. Important constraints (industry honesty)

- ≥90% is defined by **this repo's** `quality_score_v2` and gates—not arbitrary external benchmarks.
- On telemetry with **no spikes** (actual_spikes < 3), spike score defaults to neutral 0.75; quality comes from MAE/R² vs persistence.
- On **extremely short** CSV (<120 usable rows), auto_benchmark must error clearly: `"Need at least 120 rows after feature engineering"`.
- Do not leak test labels into calibration or ensemble tuning.

---

## 14. Short copy-paste prompt (minimal)

If the full prompt is too long, use:

```
Implement CODEX_ITERATIVE_BENCHMARK.md Phases A–H in AI-Model-for-Network-Traffic-main.
Build ml/auto_benchmark.py to loop train→evaluate until gates_passed all true and quality≥90
for synthetic, kaggle, and ml/telemetry.csv. Add telemetry_profile, calibrate_predictions,
trainer_tournament, run.ps1 benchmark mode. Fix evaluate_model baseline quality mismatch.
Run Section 10 verification until green; sync docs/results and docs/images.
```

---

*Spec version: 2026-05-21 — targets latest zip with metrics_utils + gates_passed already present.*

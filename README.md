# AI-Driven Network Design and Traffic Prediction

This project combines a ContainerLab spine-leaf fabric with an LSTM telemetry
pipeline. It can run fully offline with synthetic telemetry, or collect live
traffic, latency, and packet-loss data from deployed ContainerLab nodes.

## What Is Included

- ContainerLab topology: 2 spines, 4 leaves, and a telemetry utility container.
- FRRouting BGP configs for a routed spine-leaf fabric.
- Live telemetry collector that uses Docker exec and ping from the host.
- Synthetic telemetry generator for development without Docker/ContainerLab.
- Multivariate LSTM that predicts traffic, latency, and packet loss together.
- Dashboard with forecasts, errors, correlations, and spike detection.

## Training Defaults

- Lookback sequence length: `48`
- Hidden size: `128`
- LSTM layers: `2`
- Epochs: `130`
- Packet loss transform: `log1p` during training, `expm1` after prediction
- Spike-weighted loss: training quantile `0.90`, spike weight `4.0`
- Early stopping: off by default so training runs through all requested epochs
- Dataset optimized trainer: early-stopped Gradient Boosting, no spike oversampling by default

## Project Layout

```text
ai_network_project/
  containerlab/topology.clab.yml
  configs/frr/*/frr.conf
  scripts/collect_telemetry.py
  ml/generate_data.py
  ml/train_model.py
  ml/visualize.py
  run.sh
  requirements.txt
```

## Quick Start Without ContainerLab

```bash
bash run.sh synthetic --samples 720 --epochs 130
```

On Windows PowerShell, use the native runner instead:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1 synthetic -Samples 720 -Epochs 130
```

To train from the Kaggle `crawford/computer-network-traffic` dataset:

```powershell
.\run.ps1 kaggle -Samples 720 -Epochs 130
```

For best spike prediction on dataset-style telemetry, use the optimized dataset
trainer:

```powershell
.\run.ps1 kaggle_opt -Samples 5000 -SkipInstall
```

For any other telemetry dataset, save it as `ml/telemetry.csv` with columns
`timestamp`, `traffic_mbps`, `latency_ms`, and `packet_loss_pct`, then run:

```powershell
.\run.ps1 dataset_opt -SkipInstall
```

Each Kaggle run creates a new `runs/<timestamp>_kaggle/` folder and preserves
chronological order for valid time-series training. If no seed is supplied, the
loader can choose a different chronological slice each run. Optional
augmentation is available in `ml/load_kaggle_data.py`.

Every run creates its own folder under `runs/`, for example
`runs/20260519_154500_synthetic/`. The folder contains that run's
raw telemetry, model artifacts, CSV results, metrics, spike summary, and
dashboard images.

To train from an existing `ml/telemetry.csv` and store the result in a new
run folder:

```powershell
.\run.ps1 train -Epochs 130
.\run.ps1 visualize
```

Outputs are written to subfolders inside each run folder:

- `raw_data/telemetry.csv`
- `results/predictions.csv`
- `results/actuals.csv`
- `results/train_losses.csv`
- `results/evaluation_comparison.csv`
- `results/evaluation_baselines.csv`
- `results/evaluation_spikes.csv`
- `json/metrics.json`
- `json/spike_summary.json`
- `json/scaler_params.json`
- `json/evaluation_summary.json`
- `json/model_metadata.json`
- `json/model_readable_summary.json`
- `images/traffic_prediction_dashboard.png`
- `images/model_evaluation_dashboard.png`
- `model/lstm_model.pth`
- `model/dataset_model.joblib` for optimized dataset runs
- `model/model_readable_report.md`
- `model/model_weights_summary.csv`
- `model/model_gate_summary.csv`

`model/lstm_model.pth` and `model/dataset_model.joblib` are binary model files.
They will look unreadable in a text editor by design. Use
`model_readable_report.md`, `model_readable_summary.json`, `model_metadata.json`,
`metrics.json`, and the CSV files for human-readable inspection.

## Install Tools From Terminal

On Windows PowerShell, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\setup_windows.ps1
```

That installs Python, Git, Docker Desktop, WSL Ubuntu, and Python packages using
`winget` where possible. After WSL/Ubuntu is ready, open Ubuntu and run this from
the project folder:

```bash
bash scripts/setup_wsl_containerlab.sh
```

If you only want the ML pipeline and not ContainerLab, run:

```powershell
.\scripts\setup_windows.ps1 -SkipDocker -SkipWsl
```

## Live ContainerLab Workflow

ContainerLab is Linux-focused. On Windows, run this from WSL2, a Linux VM, or a
Linux host with Docker and ContainerLab installed.

From a VS Code PowerShell terminal on Windows, `run.ps1` can call WSL for the
ContainerLab deploy/destroy steps and then run the ML pipeline locally:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\run.ps1 deploy
.\run.ps1 live -Samples 120 -Interval 10 -Epochs 130
.\run.ps1 destroy
```

Use this flow after running:

```powershell
.\scripts\setup_windows.ps1
```

Then open Ubuntu once and run:

```bash
bash scripts/setup_wsl_containerlab.sh
```

The `live` mode writes ContainerLab telemetry to a new `runs/<timestamp>_live/`
folder, trains the LSTM, and creates the dashboard in that same folder.

```bash
bash run.sh deploy
bash run.sh live --samples 120 --interval 10 --epochs 130
bash run.sh destroy
```

The live collector:

1. Verifies the `clab-ai-traffic-lab-*` containers are running.
2. Reads interface byte counters from `/proc/net/dev`.
3. Sends ping probes across the leaf/spine fabric to create measurable traffic.
4. Measures latency and packet loss from the router containers.
5. Writes LSTM-ready rows to a timestamped folder under `runs/`.

## Manual Commands

```bash
python -m pip install -r requirements.txt

cd ml
python generate_data.py --hours 720 --seed 7 --output ../runs/manual/raw_data/telemetry.csv
python train_model.py --data ../runs/manual/raw_data/telemetry.csv --epochs 130 --output-dir ../runs/manual
python visualize.py --data ../runs/manual/raw_data/telemetry.csv --output-dir ../runs/manual
```

For live collection after deploying ContainerLab:

```bash
python scripts/collect_telemetry.py --mode live --samples 120 --interval 10 --output-root runs
```

## Notes

- The collector is intentionally host-run because the host has Docker access.
- Synthetic mode is still useful for demos, model iteration, and environments
  without ContainerLab.
- Dashboard spike thresholds are calculated from the training portion as
  `mean + sensitivity * std` and can be changed with
  `python visualize.py --sensitivity 2.5`.

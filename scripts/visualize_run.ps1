param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$MlDir = Join-Path $ProjectDir "ml"
$ResolvedRun = Resolve-Path $RunDir
$Telemetry = Join-Path $ResolvedRun "raw_data\telemetry.csv"

if (-not (Test-Path $Telemetry)) {
    throw "Missing raw telemetry: $Telemetry"
}

Push-Location $MlDir
python visualize.py --data $Telemetry --output-dir $ResolvedRun

$Actuals = Join-Path $ResolvedRun "results\actuals.csv"
$Predictions = Join-Path $ResolvedRun "results\predictions.csv"
$Losses = Join-Path $ResolvedRun "results\train_losses.csv"
$Metrics = Join-Path $ResolvedRun "json\metrics.json"
if ((Test-Path $Actuals) -and (Test-Path $Predictions) -and (Test-Path $Losses) -and (Test-Path $Metrics)) {
    python evaluate_model.py --run-dir $ResolvedRun
} else {
    Write-Host "Skipped evaluation dashboard because this run does not have complete prediction artifacts."
}
Pop-Location

Write-Host "Graphs:"
Write-Host "  $(Join-Path $ResolvedRun 'images\traffic_prediction_dashboard.png')"
Write-Host "  $(Join-Path $ResolvedRun 'images\model_evaluation_dashboard.png')"

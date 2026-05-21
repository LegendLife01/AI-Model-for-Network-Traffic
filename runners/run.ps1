param(
    [ValidateSet("synthetic", "kaggle", "kaggle_opt", "dataset_opt", "simulate", "live", "deploy", "destroy", "train", "visualize", "benchmark")]
    [string]$Mode = "synthetic",
    [int]$Samples = 720,
    [int]$Interval = 1,
    [int]$Epochs = 130,
    [int]$LIpn = -1,
    [double]$TargetQuality = 90,
    [int]$MaxAttempts = 24,
    [double]$MaxMinutes = -1,
    [bool]$AutoBenchmark = $true,
    [bool]$Learn = $true,
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

function Get-PythonCommand {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd -and $cmd.Source -notlike "*WindowsApps*") {
        return "python"
    }

    $cmd = Get-Command py -ErrorAction SilentlyContinue
    if ($cmd) {
        return "py"
    }

    throw "Python was not found. Install Python, then reopen this terminal."
}

$Python = Get-PythonCommand
$ArgsList = @(
    (Join-Path $ScriptDir "run.py"),
    $Mode,
    "--samples", $Samples,
    "--interval", $Interval,
    "--epochs", $Epochs,
    "--l-ipn", $LIpn,
    "--target-quality", $TargetQuality,
    "--max-attempts", $MaxAttempts
)

if ($MaxMinutes -ge 0) {
    $ArgsList += @("--max-minutes", $MaxMinutes)
}

if ($AutoBenchmark) {
    $ArgsList += "--auto-benchmark"
} else {
    $ArgsList += "--no-auto-benchmark"
}

if ($Learn) {
    $ArgsList += "--learn"
} else {
    $ArgsList += "--no-learn"
}

if ($SkipInstall) {
    $ArgsList += "--skip-install"
}

Push-Location $ProjectDir
& $Python @ArgsList
Pop-Location
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

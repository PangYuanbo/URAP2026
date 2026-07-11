param(
    [string]$Features = "U:\URAP_runs\samurai\ablation_feature_train_finetuned1\features.npz",
    [string]$Checkpoint = "U:\URAP_models\samurai\bbox_readout_finetuned1.pt",
    [string]$RunName = "ablation_bbox_readout_finetuned1",
    [int]$Epochs = 80,
    [int]$BatchSize = 1024,
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runner = Join-Path $repoRoot "tools\train_samurai_bbox_readout.py"
$controlRoot = Join-Path $repoRoot "artifacts\samurai_ablation"
New-Item -ItemType Directory -Force -Path $controlRoot, (Split-Path -Parent $Checkpoint) | Out-Null
if (-not (Test-Path $Features)) { throw "Missing features: $Features" }
$pidPath = Join-Path $controlRoot "$RunName.pid"
$metaPath = Join-Path $controlRoot "$RunName.meta.json"
$stdoutPath = Join-Path $controlRoot "$RunName.stdout.log"
$stderrPath = Join-Path $controlRoot "$RunName.stderr.log"
if (Test-Path $pidPath) {
    $oldPid = [int](Get-Content $pidPath -Raw)
    $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($old -and $old.CommandLine -like "*train_samurai_bbox_readout.py*") { throw "Already running with PID $oldPid" }
}
$arguments = @($runner, "--features", $Features, "--output", $Checkpoint, "--epochs", $Epochs,
    "--batch-size", $BatchSize, "--device", $Device)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content $pidPath -Encoding ascii
[ordered]@{
    pid = $process.Id; started_at = (Get-Date).ToString("o"); command = "$python $($arguments -join ' ')"
    features = $Features; checkpoint = $Checkpoint; epochs = $Epochs
    stdout_log = $stdoutPath; stderr_log = $stderrPath
} | ConvertTo-Json | Set-Content $metaPath -Encoding utf8
Get-Content $metaPath

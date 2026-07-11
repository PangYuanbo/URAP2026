param(
    [string]$DataYaml = "D:\URAP_local_datasets\joint_yolomg\joint_nps_ard100.yaml",
    [string]$RunDir = "C:\Users\aaron\Desktop\URAP\artifacts\joint_training\yolomg_nps_ard100_e20",
    [int]$Epochs = 20,
    [int]$BatchSize = 2,
    [int]$Workers = 2,
    [int]$ImgSz = 1280,
    [string]$Device = "0",
    [switch]$ResumeExisting
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$yolomgRoot = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG"
$python = Join-Path $yolomgRoot ".venv\Scripts\python.exe"
$weights = Join-Path $yolomgRoot "yolov5s.pt"
$config = Join-Path $yolomgRoot "models\NPS_uav_s.yaml"
$runnerRoot = Join-Path $repoRoot "artifacts\joint_training\yolomg_runner"
$logRoot = Join-Path $runnerRoot "logs"
$pidPath = Join-Path $runnerRoot "train.pid"
$metaPath = Join-Path $runnerRoot "train.meta.json"
foreach ($path in @($python, $weights, $config, $DataYaml)) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file missing: $path" } }
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = Get-Content -LiteralPath $pidPath | Select-Object -First 1
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*YOLOMG*train.py*") { Write-Output "Already running PID=$oldPid"; exit 0 }
    Write-Output "Previous training is NOT RUNNING. observed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') old_pid=$oldPid"
}

$lastCheckpoint = Join-Path $RunDir "weights\last.pt"
if ((Test-Path -LiteralPath $RunDir) -and -not $ResumeExisting) { throw "RunDir exists; use -ResumeExisting explicitly: $RunDir" }
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logRoot "train_$timestamp.out.txt"
$stderr = Join-Path $logRoot "train_$timestamp.err.txt"
if ($ResumeExisting) {
    if (-not (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)) { throw "Resume checkpoint missing: $lastCheckpoint" }
    $arguments = @("train.py", "--resume", $lastCheckpoint)
} else {
    $arguments = @(
        "train.py", "--data", $DataYaml, "--cfg", $config, "--weights", $weights,
        "--epochs", "$Epochs", "--batch-size", "$BatchSize", "--imgsz", "$ImgSz",
        "--device", $Device, "--workers", "$Workers", "--val-workers", "0", "--save-period", "1", "--project", (Split-Path $RunDir -Parent),
        "--name", (Split-Path $RunDir -Leaf), "--exist-ok"
    )
}
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $yolomgRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ASCII
@{
    pid = $process.Id; start_time = (Get-Date).ToString("o"); run_dir = $RunDir; data = $DataYaml
    epochs = $Epochs; batch_size = $BatchSize; workers = $Workers; imgsz = $ImgSz; device = $Device
    stdout = $stdout; stderr = $stderr; command = "$python $($arguments -join ' ')"
} | ConvertTo-Json | Set-Content -LiteralPath $metaPath -Encoding UTF8
Write-Output "Started detached YOLOMG joint training"
Write-Output "PID=$($process.Id)"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"

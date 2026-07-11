param(
    [string]$Config = "configs/sam2.1_training/sam2.1_hiera_b+_NPS_weakmask_finetune.yaml",
    [string]$RunName = "finetune_base_plus_nps_weakmask_v1",
    [int]$TotalEpochs = 10,
    [string]$RunRoot = "U:\URAP_runs\samurai\finetune_base_plus_nps_weakmask_v1"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$sam2Root = Join-Path $repoRoot "third_party\samurai\sam2"
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runner = Join-Path $sam2Root "training\train.py"
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
New-Item -ItemType Directory -Force -Path $controlRoot | Out-Null
$pidPath = Join-Path $controlRoot "$RunName.pid"
$metaPath = Join-Path $controlRoot "$RunName.meta.json"
$stdoutPath = Join-Path $controlRoot "$RunName.stdout.log"
$stderrPath = Join-Path $controlRoot "$RunName.stderr.log"
if (Test-Path $pidPath) {
    $oldPid = [int](Get-Content $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*training*train.py*") { throw "SAMURAI fine-tuning is already running with PID $oldPid" }
}
$arguments = @($runner, "-c", $Config, "--use-cluster", "0", "--num-gpus", "1")
$previousPythonPath = $env:PYTHONPATH
$previousPythonUtf8 = $env:PYTHONUTF8
$env:PYTHONPATH = if ($previousPythonPath) { "$sam2Root;$previousPythonPath" } else { $sam2Root }
$env:PYTHONUTF8 = "1"
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $sam2Root -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$env:PYTHONPATH = $previousPythonPath
$env:PYTHONUTF8 = $previousPythonUtf8
$process.Id | Set-Content $pidPath -Encoding ascii
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    command = "$python $($arguments -join ' ')"
    config = $Config
    total_epochs = $TotalEpochs
    run_root = $RunRoot
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content $metaPath -Encoding utf8
$metadata | ConvertTo-Json -Depth 4

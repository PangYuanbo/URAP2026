param(
    [string]$RunName = "ard100_short2_frozen_smoke",
    [double]$MemoryFraction = 0.45,
    [string]$Config = "configs/sam2.1_training/sam2.1_hiera_b+_ARD100_short2_frozen_local_smoke.yaml",
    [string]$CheckpointDir = "U:\URAP_runs\samurai\finetune_base_plus_ard100_short2_frozen_smoke\checkpoints"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
New-Item -ItemType Directory -Force -Path $controlRoot | Out-Null
$pidPath = Join-Path $controlRoot "$RunName.pid"
$metaPath = Join-Path $controlRoot "$RunName.meta.json"
$stdoutPath = Join-Path $controlRoot "$RunName.stdout.log"
$stderrPath = Join-Path $controlRoot "$RunName.stderr.log"

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*run_sam2_train_with_memory_cap.py*") {
        throw "ARD100 short2 frozen smoke is already running with PID $oldPid"
    }
}

$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "run_sam2_train_with_memory_cap.py"
$arguments = @($runner, "--memory-fraction", $MemoryFraction, "--config", $Config)
$oldPythonPath = $env:PYTHONPATH
$oldUtf8 = $env:PYTHONUTF8
$env:PYTHONPATH = Join-Path $repoRoot "third_party\samurai\sam2"
$env:PYTHONUTF8 = "1"
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory (Join-Path $repoRoot "third_party\samurai\sam2") -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$env:PYTHONPATH = $oldPythonPath
$env:PYTHONUTF8 = $oldUtf8

$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    command = "$python $($arguments -join ' ')"
    memory_fraction = $MemoryFraction
    expected_sequences = 1
    expected_frames_per_window = 2
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    config = $Config
    checkpoint_dir = $CheckpointDir
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
$metadata | ConvertTo-Json -Depth 4

param(
    [string]$DatasetRoot = "U:\URAP_datasets\ATA",
    [string]$InitialCheckpoint = "U:\URAP_models\seqtrack\ata_init\seqtrack_b384_template192_init.pth.tar",
    [string]$RunRoot = "U:\URAP_runs\ata\seqtrack_b384_ata_50ep"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$seqtrackRoot = Join-Path $repoRoot "third_party\VideoX\SeqTrack"
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$controlRoot = Join-Path $repoRoot "artifacts\ata_reproduction\seqtrack_train"
New-Item -ItemType Directory -Force -Path $controlRoot, $RunRoot | Out-Null
if (-not (Test-Path $InitialCheckpoint)) { throw "Missing initialization checkpoint: $InitialCheckpoint" }
$ready = & $python -c "import sys; sys.path.insert(0, sys.argv[2]); from qstr_dronedet.ata_benchmark import audit_ata_release; print(audit_ata_release(sys.argv[1])['ready_for_tracking'])" $DatasetRoot $repoRoot
if ($LASTEXITCODE -ne 0 -or $ready.Trim() -ne "True") { throw "ATA dataset is incomplete; training was not started" }

$pidPath = Join-Path $controlRoot "run.pid"
$metaPath = Join-Path $controlRoot "run.meta.json"
$stdoutPath = Join-Path $controlRoot "run.stdout.log"
$stderrPath = Join-Path $controlRoot "run.stderr.log"
if (Test-Path $pidPath) {
    $oldPid = [int](Get-Content $pidPath -Raw)
    $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($old -and $old.CommandLine -like "*run_training.py*seqtrack_b384_ata*") { throw "Already running with PID $oldPid" }
}

$env:ATA_DATASET_ROOT = $DatasetRoot
$env:SEQTRACK_PRETRAINED_CHECKPOINT = $InitialCheckpoint
$env:TORCH_HOME = "U:\URAP_models\torch_cache"
$arguments = @("lib/train/run_training.py", "--script", "seqtrack", "--config", "seqtrack_b384_ata",
    "--save_dir", $RunRoot, "--use_lmdb", "0", "--seed", "20260624")
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $seqtrackRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content $pidPath -Encoding ascii
[ordered]@{
    pid = $process.Id; started_at = (Get-Date).ToString("o"); command = "$python $($arguments -join ' ')"
    dataset_root = $DatasetRoot; initial_checkpoint = $InitialCheckpoint; run_root = $RunRoot
    stdout_log = $stdoutPath; stderr_log = $stderrPath
} | ConvertTo-Json | Set-Content $metaPath -Encoding utf8
Get-Content $metaPath

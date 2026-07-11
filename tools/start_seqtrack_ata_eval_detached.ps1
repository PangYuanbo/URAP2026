param(
    [string]$DatasetRoot = "U:\URAP_datasets\ATA",
    [string]$Checkpoint = "U:\URAP_models\seqtrack\train\seqtrack\seqtrack_b384\SEQTRACK_ep0500.pth.tar",
    [string]$RunRoot = "U:\URAP_runs\ata\seqtrack_b384_zero_shot"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$runner = Join-Path $repoRoot "tools\run_seqtrack_ata.py"
$seqtrackRoot = Join-Path $repoRoot "third_party\VideoX\SeqTrack"
$config = Join-Path $seqtrackRoot "experiments\seqtrack\seqtrack_b384.yaml"
$controlRoot = Join-Path $repoRoot "artifacts\ata_reproduction\seqtrack_eval"
New-Item -ItemType Directory -Force -Path $controlRoot, $RunRoot | Out-Null
$ready = & $python -c "import sys; sys.path.insert(0, sys.argv[2]); from qstr_dronedet.ata_benchmark import audit_ata_release; print(audit_ata_release(sys.argv[1])['ready_for_tracking'])" $DatasetRoot $repoRoot
if ($LASTEXITCODE -ne 0 -or $ready.Trim() -ne "True") { throw "ATA dataset is incomplete; evaluation was not started" }
$pidPath = Join-Path $controlRoot "run.pid"
$metaPath = Join-Path $controlRoot "run.meta.json"
$stdoutPath = Join-Path $controlRoot "run.stdout.log"
$stderrPath = Join-Path $controlRoot "run.stderr.log"

if (Test-Path $pidPath) {
    $oldPid = [int](Get-Content $pidPath -Raw)
    $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($old -and $old.CommandLine -like "*run_seqtrack_ata.py*") { throw "Already running with PID $oldPid" }
}

$arguments = @($runner, "--dataset-root", $DatasetRoot, "--seqtrack-root", $seqtrackRoot,
    "--checkpoint", $Checkpoint, "--config", $config, "--output-root", $RunRoot, "--split", "test")
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden `
    -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content $pidPath -Encoding ascii
[ordered]@{
    pid = $process.Id; started_at = (Get-Date).ToString("o"); command = "$python $($arguments -join ' ')"
    run_root = $RunRoot; stdout_log = $stdoutPath; stderr_log = $stderrPath
    progress_file = (Join-Path $RunRoot "progress.json")
} | ConvertTo-Json | Set-Content $metaPath -Encoding utf8
Get-Content $metaPath

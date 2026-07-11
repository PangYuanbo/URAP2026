param(
    [string]$DatasetRoot = "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI_SHORT166",
    [string]$RunRoot = "U:\URAP_runs\samurai\ard100_short166_experiment_v1",
    [int]$MinimumFreeMiB = 29000
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\ard100_short166_local_experiment_v1"
$pidPath = Join-Path $controlRoot "orchestrator.pid"
$metaPath = Join-Path $controlRoot "orchestrator.meta.json"
$stdoutPath = Join-Path $controlRoot "orchestrator.stdout.log"
$stderrPath = Join-Path $controlRoot "orchestrator.stderr.log"
New-Item -ItemType Directory -Force -Path $controlRoot | Out-Null

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*sequence_ard100_short166_local_experiment.ps1*") {
        throw "ARD100 short166 local experiment is already running with PID $oldPid"
    }
}

$worker = Join-Path $PSScriptRoot "sequence_ard100_short166_local_experiment.ps1"
$arguments = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $worker,
    "-DatasetRoot", $DatasetRoot,
    "-RunRoot", $RunRoot,
    "-MinimumFreeMiB", $MinimumFreeMiB
)
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    command = "powershell.exe $($arguments -join ' ')"
    dataset_root = $DatasetRoot
    run_root = $RunRoot
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
$metadata | ConvertTo-Json -Depth 4

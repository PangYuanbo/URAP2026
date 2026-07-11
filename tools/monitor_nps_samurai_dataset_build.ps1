param([ValidateSet("train", "val", "test")][string]$Split = "train")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$runName = "dataset_build_${Split}_v1"
$pidPath = Join-Path $controlRoot "$runName.pid"
$metaPath = Join-Path $controlRoot "$runName.meta.json"
if (-not (Test-Path $metaPath)) { throw "Missing metadata: $metaPath" }
$meta = Get-Content $metaPath -Raw | ConvertFrom-Json
$pidValue = if (Test-Path $pidPath) { [int](Get-Content $pidPath -Raw) } else { [int]$meta.pid }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$commandMatches = $process -and $process.CommandLine -like "*build_nps_samurai_dataset.py*"
$progress = if (Test-Path $meta.progress_file) { Get-Content $meta.progress_file -Raw | ConvertFrom-Json } else { $null }
$lastOutput = Get-ChildItem $meta.stdout_log, $meta.stderr_log, $meta.progress_file -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
[ordered]@{
    status = if ($commandMatches) { "RUNNING" } else { "NOT RUNNING" }
    done_total = if ($progress) { "$($progress.done_sequences)/$($progress.total_sequences)" } else { "0/?" }
    pid = $pidValue
    start_time = $meta.started_at
    command_matches = [bool]$commandMatches
    last_completed_unit = if ($progress) { $progress.last_completed_sequence } else { $null }
    done_frames = if ($progress) { $progress.done_frames } else { 0 }
    last_output_timestamp = if ($lastOutput) { $lastOutput.LastWriteTime.ToString("o") } else { $null }
    progress_status = if ($progress) { $progress.status } else { "not-created" }
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
    progress_file = $meta.progress_file
} | ConvertTo-Json -Depth 4

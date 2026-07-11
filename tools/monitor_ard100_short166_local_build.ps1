param([string]$RunName = "ard100_short166_local_build_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$root = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$meta = Get-Content -LiteralPath (Join-Path $root "$RunName.meta.json") -Raw | ConvertFrom-Json
$statePath = Join-Path $root "state.json"
$state = if (Test-Path -LiteralPath $statePath) { Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json } else { $null }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
$matches = $process -and $process.CommandLine -like "*sequence_ard100_short166_local_build.ps1*"
$child = if ($state -and $state.child_pid) { Get-CimInstance Win32_Process -Filter "ProcessId=$($state.child_pid)" -ErrorAction SilentlyContinue } else { $null }
$progress = $null
if ($state -and $state.split) {
    $progressPath = "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI_SHORT166\$($state.split)_v1\progress.json"
    if (Test-Path -LiteralPath $progressPath) { $progress = Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json }
}
$materializeProgressPath = "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI_SHORT166\materialize_progress.json"
$materializeProgress = if (Test-Path -LiteralPath $materializeProgressPath) { Get-Content -LiteralPath $materializeProgressPath -Raw | ConvertFrom-Json } else { $null }
$logCandidates = @($meta.stdout_log, $meta.stderr_log, $statePath)
if ($state -and $state.stdout_log) { $logCandidates += $state.stdout_log }
if ($state -and $state.stderr_log) { $logCandidates += $state.stderr_log }
if (Test-Path -LiteralPath $materializeProgressPath) { $logCandidates += $materializeProgressPath }
$lastFile = Get-ChildItem $logCandidates -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
[ordered]@{
    status = if ($matches) { "RUNNING" } elseif ($state -and $state.stage -eq "completed") { "NOT RUNNING (COMPLETED)" } else { "NOT RUNNING" }
    done_total = if ($state) { "$($state.done)/$($state.total) splits" } else { "0/3 splits" }
    pid = [int]$meta.pid
    start_time = $meta.started_at
    command_matches = [bool]$matches
    stage = if ($state) { $state.stage } else { "initializing" }
    child_pid = if ($state) { $state.child_pid } else { $null }
    child_command_matches = [bool]($child -and $child.CommandLine -like "*build_ard100_short_tracklets.py*")
    sequence_done_total = if ($progress) { "$($progress.done_sequences)/$($progress.total_sequences)" } elseif ($materializeProgress) { "$($materializeProgress.checked_frames) frames verified" } else { $null }
    last_completed_unit = if ($progress) { $progress.last_completed_sequence } elseif ($materializeProgress -and $materializeProgress.sequence) { "$($materializeProgress.split)/$($materializeProgress.sequence)" } elseif ($state) { $state.last_completed_unit } else { $null }
    last_output_timestamp = if ($lastFile) { $lastFile.LastWriteTime.ToString("o") } else { $meta.started_at }
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
    child_stdout_log = if ($state) { $state.stdout_log } else { $null }
    child_stderr_log = if ($state) { $state.stderr_log } else { $null }
} | ConvertTo-Json -Depth 6

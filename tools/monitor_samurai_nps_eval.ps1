param(
    [string]$RunName = "zero_shot_tiny_val_v1"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$pidPath = Join-Path $controlRoot "$RunName.pid"
$metaPath = Join-Path $controlRoot "$RunName.meta.json"
if (-not (Test-Path $metaPath)) { throw "Missing metadata: $metaPath" }
$meta = Get-Content $metaPath -Raw | ConvertFrom-Json
$pidValue = if (Test-Path $pidPath) { [int](Get-Content $pidPath -Raw) } else { [int]$meta.pid }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$commandMatches = $process -and $process.Name -eq "python.exe" -and $process.CommandLine -like "*eval_samurai_nps.py*"
$children = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.ParentProcessId -eq $pidValue -and $_.Name -eq "python.exe" -and $_.CommandLine -like "*eval_samurai_nps.py*" })
$computePids = @($pidValue) + @($children | Select-Object -ExpandProperty ProcessId)
$computeProcesses = @(Get-Process -Id $computePids -ErrorAction SilentlyContinue)
$cpuSeconds = ($computeProcesses | Measure-Object CPU -Sum).Sum
$status = if ($commandMatches) { "RUNNING" } else { "NOT RUNNING" }
$progress = if (Test-Path $meta.progress_file) { Get-Content $meta.progress_file -Raw | ConvertFrom-Json } else { $null }
$lastOutput = Get-ChildItem $meta.stdout_log, $meta.stderr_log, $meta.progress_file -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null

[ordered]@{
    status = $status
    done_total = if ($progress) { "$($progress.done_sequences)/$($progress.total_sequences)" } else { "0/?" }
    pid = $pidValue
    start_time = $meta.started_at
    command_matches = [bool]$commandMatches
    compute_pids = $computePids
    cpu_seconds = if ($null -ne $cpuSeconds) { [math]::Round($cpuSeconds, 1) } else { 0 }
    last_completed_unit = if ($progress.last_completed_sequence) { $progress.last_completed_sequence } elseif ($progress.last_sequence) { "$($progress.last_sequence):frame=$($progress.last_frame)" } else { $null }
    done_frames = if ($progress) { $progress.done_frames } else { 0 }
    last_output_timestamp = if ($lastOutput) { $lastOutput.LastWriteTime.ToString("o") } else { $null }
    progress_status = if ($progress) { $progress.status } else { "not-created" }
    gpu_utilization_memory_mib = $gpu
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
    progress_file = $meta.progress_file
} | ConvertTo-Json -Depth 4

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\ata_reproduction\seqtrack_eval"
$metaPath = Join-Path $controlRoot "run.meta.json"
$pidPath = Join-Path $controlRoot "run.pid"
if (-not (Test-Path $metaPath)) { throw "Missing metadata: $metaPath" }
$meta = Get-Content $metaPath -Raw | ConvertFrom-Json
$pidValue = if (Test-Path $pidPath) { [int](Get-Content $pidPath -Raw) } else { [int]$meta.pid }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$commandMatches = $process -and $process.CommandLine -like "*run_seqtrack_ata.py*"
$progress = if (Test-Path $meta.progress_file) { Get-Content $meta.progress_file -Raw | ConvertFrom-Json } else { $null }
$files = @($meta.stdout_log, $meta.stderr_log, $meta.progress_file) | Where-Object { Test-Path $_ }
$lastOutput = $files | ForEach-Object { Get-Item $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
[ordered]@{
    status = if ($commandMatches) { "RUNNING" } else { "NOT RUNNING" }
    done_total = if ($progress) { "$($progress.done_sequences)/$($progress.total_sequences)" } else { "0/10" }
    pid = $pidValue; start_time = $meta.started_at; command_matches = [bool]$commandMatches
    last_completed_unit = if ($progress.last_completed_sequence) { $progress.last_completed_sequence } else { $progress.last_sequence }
    done_frames = if ($progress) { $progress.done_frames } else { 0 }
    last_frame = if ($progress) { $progress.last_frame } else { 0 }
    last_output_timestamp = if ($lastOutput) { $lastOutput.LastWriteTime.ToString("o") } else { $null }
    progress_status = if ($progress) { $progress.status } else { "not-created" }
    gpu_utilization_memory_mib = $gpu; stdout_log = $meta.stdout_log; stderr_log = $meta.stderr_log
    progress_file = $meta.progress_file
} | ConvertTo-Json -Depth 4

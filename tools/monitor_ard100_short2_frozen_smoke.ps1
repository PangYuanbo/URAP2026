param([string]$RunName = "ard100_short2_frozen_smoke")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$meta = Get-Content -LiteralPath (Join-Path $controlRoot "$RunName.meta.json") -Raw | ConvertFrom-Json
$pidValue = [int](Get-Content -LiteralPath (Join-Path $controlRoot "$RunName.pid") -Raw)
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$matches = $process -and $process.CommandLine -like "*run_sam2_train_with_memory_cap.py*"
$stdoutItem = Get-Item -LiteralPath $meta.stdout_log -ErrorAction SilentlyContinue
$stderrItem = Get-Item -LiteralPath $meta.stderr_log -ErrorAction SilentlyContinue
$lines = @()
if ($stdoutItem) { $lines += Get-Content -LiteralPath $meta.stdout_log -Tail 120 }
if ($stderrItem) { $lines += Get-Content -LiteralPath $meta.stderr_log -Tail 120 }
$progress = @($lines | Select-String -Pattern 'Train Epoch:|Mem \(GB\)|OutOfMemoryError|CUDA out of memory|Traceback' | ForEach-Object { $_.Line } | Select-Object -Last 12)
$checkpoint = Join-Path $meta.checkpoint_dir "checkpoint.pt"
$completed = Test-Path -LiteralPath $checkpoint -PathType Leaf
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits 2>$null

[ordered]@{
    status = if ($matches) { "RUNNING" } elseif ($completed) { "NOT RUNNING (COMPLETED)" } else { "NOT RUNNING" }
    done_total = if ($completed) { "1/1 sequence smoke" } else { "0/1 sequence smoke" }
    pid = $pidValue
    start_time = $meta.started_at
    command_matches = [bool]$matches
    memory_fraction = $meta.memory_fraction
    last_completed_unit = if ($completed) { "checkpoint.pt" } elseif ($progress.Count) { $progress[-1] } else { "initializing" }
    last_output_timestamp = if ($stdoutItem -or $stderrItem) { @($stdoutItem, $stderrItem) | Where-Object { $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty LastWriteTime | ForEach-Object { $_.ToString("o") } } else { $null }
    gpu_utilization_used_free_mib = $gpu
    checkpoint = if ($completed) { $checkpoint } else { $null }
    recent_progress = $progress
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
} | ConvertTo-Json -Depth 5

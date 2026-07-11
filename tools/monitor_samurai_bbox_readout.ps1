param([string]$RunName = "ablation_bbox_readout_finetuned1")
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_ablation"
$meta = Get-Content (Join-Path $controlRoot "$RunName.meta.json") -Raw | ConvertFrom-Json
$pidValue = [int]$meta.pid
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$processMatches = $process -and $process.Name -eq "python.exe" -and $process.CommandLine -like "*train_samurai_bbox_readout.py*"
$epochRows = if (Test-Path $meta.stdout_log) { Get-Content $meta.stdout_log | Where-Object { $_ -match '^\{"epoch"' } } else { @() }
$checkpointComplete = (Test-Path $meta.checkpoint) -and $epochRows.Count -eq [int]$meta.epochs
$lastOutput = Get-ChildItem $meta.stdout_log, $meta.stderr_log, $meta.checkpoint -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
[ordered]@{
    status = if ($processMatches) { "RUNNING" } elseif ($checkpointComplete) { "COMPLETE" } else { "NOT RUNNING" }
    done_total = "$($epochRows.Count)/$($meta.epochs)"
    pid = $pidValue; start_time = $meta.started_at
    last_completed_unit = if ($epochRows.Count) { ($epochRows[-1] | ConvertFrom-Json).epoch } else { $null }
    last_output_timestamp = if ($lastOutput) { $lastOutput.LastWriteTime.ToString("o") } else { $null }
    gpu_utilization_memory_mib = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null)
    stdout_log = $meta.stdout_log; stderr_log = $meta.stderr_log; checkpoint = $meta.checkpoint
} | ConvertTo-Json -Depth 4

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\ata_reproduction\seqtrack_train"
$metaPath = Join-Path $controlRoot "run.meta.json"
$pidPath = Join-Path $controlRoot "run.pid"
if (-not (Test-Path $metaPath)) { throw "Missing metadata: $metaPath" }
$meta = Get-Content $metaPath -Raw | ConvertFrom-Json
$pidValue = if (Test-Path $pidPath) { [int](Get-Content $pidPath -Raw) } else { [int]$meta.pid }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$commandMatches = $process -and $process.CommandLine -like "*run_training.py*seqtrack_b384_ata*"
$checkpoints = Get-ChildItem (Join-Path $meta.run_root "checkpoints\train\seqtrack\seqtrack_b384_ata\SEQTRACK_ep*.pth.tar") -ErrorAction SilentlyContinue | Sort-Object Name
$lastCheckpoint = $checkpoints | Select-Object -Last 1
$lastOutput = Get-ChildItem $meta.stdout_log, $meta.stderr_log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$lastLogLine = if (Test-Path $meta.stdout_log) { Get-Content $meta.stdout_log -Tail 1 } else { $null }
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null
[ordered]@{
    status = if ($commandMatches) { "RUNNING" } else { "NOT RUNNING" }
    done_total = "$($checkpoints.Count)/50"
    pid = $pidValue; start_time = $meta.started_at; command_matches = [bool]$commandMatches
    last_completed_unit = if ($lastCheckpoint) { $lastCheckpoint.BaseName } else { $null }
    last_output_timestamp = if ($lastOutput) { $lastOutput.LastWriteTime.ToString("o") } else { $null }
    last_log_line = $lastLogLine; gpu_utilization_memory_mib = $gpu
    stdout_log = $meta.stdout_log; stderr_log = $meta.stderr_log
} | ConvertTo-Json -Depth 4

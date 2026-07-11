$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\ard100_short166_local_experiment_v1"
$pidPath = Join-Path $controlRoot "orchestrator.pid"
$metaPath = Join-Path $controlRoot "orchestrator.meta.json"
$statePath = Join-Path $controlRoot "state.json"
if (-not (Test-Path -LiteralPath $metaPath)) { throw "Missing metadata: $metaPath" }
$meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
$pidValue = if (Test-Path -LiteralPath $pidPath) { [int](Get-Content -LiteralPath $pidPath -Raw) } else { [int]$meta.pid }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$commandMatches = $process -and $process.CommandLine -like "*sequence_ard100_short166_local_experiment.ps1*"
$state = if (Test-Path -LiteralPath $statePath) { Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json } else { $null }
$lastFile = Get-ChildItem $meta.stdout_log, $meta.stderr_log, $statePath -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits 2>$null
[ordered]@{
    status = if ($commandMatches) { "RUNNING" } elseif ($state -and $state.stage -eq "completed") { "NOT RUNNING (COMPLETED)" } else { "NOT RUNNING" }
    done_total = if ($state) { "$($state.done)/$($state.total)" } else { "0/6" }
    pid = $pidValue
    start_time = $meta.started_at
    command_matches = [bool]$commandMatches
    stage = if ($state) { $state.stage } else { "initializing" }
    last_completed_unit = if ($state) { $state.last_completed_unit } else { $null }
    last_output_timestamp = if ($lastFile) { $lastFile.LastWriteTime.ToString("o") } else { $null }
    child_pid = if ($state) { $state.child_pid } else { $null }
    gpu_utilization_used_free_mib = $gpu
    state = $state
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
} | ConvertTo-Json -Depth 8

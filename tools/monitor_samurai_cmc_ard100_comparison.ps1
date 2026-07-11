param([string]$RunName = "samurai_cmc_ard100_comparison")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$meta = Get-Content -LiteralPath (Join-Path $controlRoot "$RunName.meta.json") -Raw | ConvertFrom-Json
$pidValue = [int](Get-Content -LiteralPath (Join-Path $controlRoot "$RunName.pid") -Raw)
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$matches = $process -and $process.CommandLine -like "*sequence_samurai_cmc_ard100_comparison.ps1*"
$state = if (Test-Path -LiteralPath $meta.state_file) { Get-Content -LiteralPath $meta.state_file -Raw | ConvertFrom-Json } else { $null }
$lastOutput = Get-ChildItem $meta.stdout_log, $meta.stderr_log, $meta.state_file -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits 2>$null
[ordered]@{
    status = if ($matches) { "RUNNING" } elseif ($state.status -eq "completed") { "NOT RUNNING (COMPLETED)" } else { "NOT RUNNING" }
    done_total = if ($state) { "$($state.done)/$($state.total)" } else { "0/2" }
    pid = $pidValue
    start_time = $meta.started_at
    command_matches = [bool]$matches
    current_phase = if ($state) { $state.status } else { "initializing" }
    run = if ($state.run) { $state.run } else { $null }
    child_pid = if ($state.child_pid) { $state.child_pid } elseif ($state.upstream_pid) { $state.upstream_pid } else { $null }
    evaluation_done_total = if ($null -ne $state.evaluation_done) { "$($state.evaluation_done)/$($state.evaluation_total)" } else { $null }
    last_completed_unit = if ($state.last_sequence) { $state.last_sequence } else { $null }
    last_output_timestamp = if ($lastOutput) { $lastOutput.LastWriteTime.ToString("o") } else { $null }
    gpu_utilization_used_free_mib = $gpu
    state_file = $meta.state_file
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
    error = if ($state.error) { $state.error } else { $null }
} | ConvertTo-Json -Depth 6

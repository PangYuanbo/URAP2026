param([string]$RunName = "ard100_short166_experiment_v1")

$repoRoot = Split-Path -Parent $PSScriptRoot
$root = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$meta = Get-Content -LiteralPath (Join-Path $root "$RunName.meta.json") -Raw | ConvertFrom-Json
$statePath = Join-Path $root "state.json"
$state = if (Test-Path -LiteralPath $statePath) { Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json } else { $null }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
$matches = $process -and $process.CommandLine -like "*sequence_ard100_short166_experiment.ps1*"
$stderr = Get-Item -LiteralPath $meta.stderr_log -ErrorAction SilentlyContinue
[ordered]@{
    status = if ($matches) { "RUNNING" } else { "NOT RUNNING" }
    pid = [int]$meta.pid
    started_at = $meta.started_at
    command_line_matches = [bool]$matches
    stage = if ($state) { $state.stage } else { "initializing" }
    done = if ($state -and $null -ne $state.done) { $state.done } else { 0 }
    total = if ($state -and $null -ne $state.total) { $state.total } else { 3 }
    last_output_timestamp = if ($state) { $state.updated_at } else { $meta.started_at }
    state = $state
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
    stderr_bytes = if ($stderr) { $stderr.Length } else { 0 }
} | ConvertTo-Json -Depth 8

param([string]$RunName = "ard100_short166_train_smoke_v1")

$repoRoot = Split-Path -Parent $PSScriptRoot
$root = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$meta = Get-Content -LiteralPath (Join-Path $root "$RunName.meta.json") -Raw | ConvertFrom-Json
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
$matches = $process -and $process.CommandLine -like "*modal_train_ard100_short166.py*smoke-only*"
$stdoutItem = Get-Item -LiteralPath $meta.stdout_log -ErrorAction SilentlyContinue
$stderrItem = Get-Item -LiteralPath $meta.stderr_log -ErrorAction SilentlyContinue
$lines = if ($stdoutItem) { @(Select-String -LiteralPath $meta.stdout_log -Pattern 'Train Epoch|checkpoint|"status": "completed"|App completed' | Select-Object -Last 12 | ForEach-Object { $_.Line }) } else { @() }
[ordered]@{
    status = if ($matches) { "RUNNING" } else { "NOT RUNNING" }
    pid = [int]$meta.pid
    started_at = $meta.started_at
    command_line_matches = [bool]$matches
    done = if ($lines -match '"status": "completed"') { 1 } else { 0 }
    total = 1
    last_output_timestamp = if ($stdoutItem) { $stdoutItem.LastWriteTime.ToString("o") } else { $meta.started_at }
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
    stderr_bytes = if ($stderrItem) { $stderrItem.Length } else { 0 }
    recent_progress = $lines
} | ConvertTo-Json -Depth 5

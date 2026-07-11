param(
    [ValidateSet("smoke", "train")]
    [string]$Mode = "smoke",
    [string]$RunName = "ard100_short166_modal_train_v1"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$pidPath = Join-Path $controlRoot "$Mode.pid"
$metaPath = Join-Path $controlRoot "$Mode.meta.json"
if (-not (Test-Path -LiteralPath $metaPath)) { throw "Missing metadata: $metaPath" }

$meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
$pidValue = if (Test-Path -LiteralPath $pidPath) { [int](Get-Content -LiteralPath $pidPath -Raw) } else { [int]$meta.pid }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
$commandMatches = $process -and $process.CommandLine -like "*modal_train_ard100_short166.py*"
$stdoutItem = Get-Item -LiteralPath $meta.stdout_log -ErrorAction SilentlyContinue
$stderrItem = Get-Item -LiteralPath $meta.stderr_log -ErrorAction SilentlyContinue
$stdoutTail = if ($stdoutItem) { @(Get-Content -LiteralPath $meta.stdout_log -Tail 80) } else { @() }
$completed = [bool]($stdoutTail -match 'App completed' -and $stdoutTail -match '"status"\s*:\s*"completed"')
$failed = [bool](($stderrItem -and $stderrItem.Length -gt 0) -or $stdoutTail -match 'Traceback|Error:|Workspace is not eligible')
$progress = @($stdoutTail | Select-String -Pattern 'Train Epoch:|results_commit|"checkpoint"|Workspace is not eligible|Error:' | ForEach-Object { $_.Line } | Select-Object -Last 12)
$lastOutput = @($stdoutItem, $stderrItem) | Where-Object { $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1

[ordered]@{
    status = if ($commandMatches) { "RUNNING" } elseif ($completed) { "NOT RUNNING (COMPLETED)" } else { "NOT RUNNING" }
    done_total = if ($completed) { "1/1" } else { "0/1" }
    pid = $pidValue
    start_time = $meta.started_at
    command_matches = [bool]$commandMatches
    mode = $Mode
    last_completed_unit = if ($completed) { $Mode } elseif ($failed) { "failed" } else { "initializing_or_running" }
    last_output_timestamp = if ($lastOutput) { $lastOutput.LastWriteTime.ToString("o") } else { $null }
    stderr_bytes = if ($stderrItem) { $stderrItem.Length } else { 0 }
    recent_progress = $progress
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
} | ConvertTo-Json -Depth 5

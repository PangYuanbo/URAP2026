param([string]$RunId = 'aot_action_chunk_transfer_v1', [int]$TailLines = 20)
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$run = Join-Path $repo "artifacts\route_b_official\${RunId}_runner"
$metaPath = Join-Path $run 'meta.json'
if (-not (Test-Path $metaPath)) { Write-Host 'NOT RUNNING: no meta file'; exit 1 }
$meta = Get-Content -Raw $metaPath | ConvertFrom-Json
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
$alive = [bool]($process -and $process.CommandLine -like '*run_aot_action_chunk_transfer.ps1*')
$progress = if (Test-Path $meta.progress) { Get-Content -Raw $meta.progress | ConvertFrom-Json } else { $null }
if ($alive) { Write-Host "RUNNING PID=$($meta.pid) START=$($meta.start_time)" } else { Write-Host "NOT RUNNING PID=$($meta.pid) START=$($meta.start_time)" }
if ($progress) { Write-Host "done/total: $($progress.done)/$($progress.total)"; Write-Host "last completed unit: $($progress.phase)"; Write-Host "last output timestamp: $($progress.updated)" } else { Write-Host 'done/total: 0/6'; Write-Host 'last completed unit: none'; Write-Host 'last output timestamp: missing' }
Write-Host "stdout log: $($meta.stdout_log)"
Write-Host "stderr log: $($meta.stderr_log)"
if (Test-Path $meta.stdout_log) { Get-Content -Tail $TailLines $meta.stdout_log }
if (Test-Path $meta.stderr_log) { Get-Content -Tail $TailLines $meta.stderr_log }

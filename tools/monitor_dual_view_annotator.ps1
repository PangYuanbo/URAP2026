$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$latestFile = Join-Path $repo 'artifacts\runs\dual_view_annotator_latest.json'
if (-not (Test-Path -LiteralPath $latestFile)) { throw "No annotator run pointer: $latestFile" }
$latest = Get-Content -LiteralPath $latestFile -Raw | ConvertFrom-Json
$processId = [int]$latest.pid
$process = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
$listener = Get-NetTCPConnection -State Listen -LocalPort ([int]$latest.port) -ErrorAction SilentlyContinue
Write-Output "Server: $(if ($process -and $listener) { 'RUNNING' } else { 'NOT RUNNING' })"
Write-Output "PID: $processId"
Write-Output "Start time: $($latest.started_at)"
if ($process) { Write-Output "Command: $($process.CommandLine)" }
Write-Output "URL: $($latest.url)"
Write-Output "Logs: $($latest.stdout_log) ; $($latest.stderr_log)"
if (Test-Path -LiteralPath $latest.stderr_log) { $tail=Get-Content -LiteralPath $latest.stderr_log -Tail 8; if($tail){Write-Output '--- request log ---';$tail} }

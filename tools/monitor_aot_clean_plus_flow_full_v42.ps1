param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId = 'aot_clean_plus_flow_full_v42',
  [string]$OutputRoot = ''
)

if (-not $OutputRoot) {
  $OutputRoot = Join-Path $RepoRoot 'artifacts\route_b_official\aot_clean_plus_flow_full_v42_runner'
}
$pidFile = Join-Path $OutputRoot "${RunId}_pid.txt"
$metaFile = Join-Path $OutputRoot "${RunId}_meta.txt"
$progressFile = Join-Path $OutputRoot "${RunId}_progress.json"
$jobPid = if (Test-Path -LiteralPath $pidFile) { Get-Content -LiteralPath $pidFile | Select-Object -First 1 } else { '' }
$process = if ($jobPid -match '^\d+$') { Get-CimInstance Win32_Process -Filter "ProcessId = $jobPid" -ErrorAction SilentlyContinue } else { $null }
$progress = if (Test-Path -LiteralPath $progressFile) { Get-Content -LiteralPath $progressFile -Raw | ConvertFrom-Json } else { $null }
$logs = Get-ChildItem (Join-Path $OutputRoot 'logs') -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
[pscustomobject]@{
  Status = if ($process) { 'RUNNING' } elseif ($progress -and $progress.status -eq 'complete') { 'COMPLETE / NOT RUNNING' } else { 'NOT RUNNING' }
  Done = if ($progress) { $progress.done } else { 0 }
  Total = if ($progress) { $progress.total } else { 18 }
  PID = $jobPid
  Started = if (Test-Path -LiteralPath $metaFile) { (Get-Content -LiteralPath $metaFile | Where-Object { $_ -like 'started=*' }) -replace '^started=', '' } else { '' }
  LastCompletedUnit = if ($progress) { $progress.last_completed_unit } else { '' }
  LastProgressTimestamp = if (Test-Path -LiteralPath $progressFile) { (Get-Item -LiteralPath $progressFile).LastWriteTime } else { $null }
  LatestLog = if ($logs) { $logs[0].FullName } else { '' }
  LatestLogTimestamp = if ($logs) { $logs[0].LastWriteTime } else { $null }
  ProgressFile = $progressFile
} | Format-List
if ($progress) {
  $progress | ConvertTo-Json -Depth 8
}

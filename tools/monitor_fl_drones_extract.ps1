param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId = 'fl_drones_extract',
  [string]$OutputRoot = ''
)
$ErrorActionPreference = 'Stop'
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot 'artifacts\benchmarks\fl_drones_extract' }
$pidFile = Join-Path $OutputRoot ("{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("{0}_meta.txt" -f $RunId)
$progressFile = Join-Path $OutputRoot ("{0}_progress.json" -f $RunId)
$pidValue = if (Test-Path -LiteralPath $pidFile) { Get-Content -LiteralPath $pidFile | Select-Object -First 1 } else { '' }
$process = if ($pidValue -match '^\d+$') { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
if ($process -and $process.CommandLine -like '*extract_zip_with_progress.py*') {
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$([Management.ManagementDateTimeConverter]::ToDateTime($process.CreationDate).ToString('yyyy-MM-dd HH:mm:ss'))"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else {
  Write-Host "NOT RUNNING PID=$pidValue"
}
if (Test-Path -LiteralPath $progressFile) {
  $progress = Get-Content -LiteralPath $progressFile -Raw | ConvertFrom-Json
  Write-Host "done/total: $($progress.done)/$($progress.total)"
  Write-Host "bytes: $($progress.bytes_done)/$($progress.bytes_total)"
  Write-Host "last output timestamp: $($progress.last_output_timestamp)"
  Write-Host "last completed unit: $($progress.last_completed_unit)"
  Write-Host "status: $($progress.status)"
} else {
  Write-Host 'done/total: 0/12'
  Write-Host 'last output timestamp: none'
  Write-Host 'last completed unit: none'
}
if (Test-Path -LiteralPath $metaFile) {
  $meta = Get-Content -LiteralPath $metaFile
  $meta | Where-Object { $_ -like 'stdout=*' -or $_ -like 'stderr=*' }
}

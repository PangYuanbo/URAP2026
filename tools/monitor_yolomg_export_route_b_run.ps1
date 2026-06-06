param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\yolomg_action\route_b_export_detached',
  [string]$RunId = 'yolomg_export_route_b_run',
  [int]$TotalImages = 0
)

$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)
if (-not (Test-Path -Path $pidFile -PathType Leaf)) {
  Write-Host "NOT RUNNING: pid file missing: $pidFile"
  exit 0
}

$pidText = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
$proc = $null
if ($pidText -match '^\d+$') {
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $pidText" -ErrorAction SilentlyContinue
}

$meta = @{}
if (Test-Path -Path $metaFile -PathType Leaf) {
  foreach ($line in Get-Content $metaFile) {
    $parts = $line -split '=', 2
    if ($parts.Count -eq 2) { $meta[$parts[0]] = $parts[1] }
  }
}
$stdout = if ($meta.ContainsKey('stdout')) { [string]$meta['stdout'] } else { '' }
$stderr = if ($meta.ContainsKey('stderr')) { [string]$meta['stderr'] } else { '' }
$outRunRoot = if ($meta.ContainsKey('out_run_root')) { [string]$meta['out_run_root'] } else { '' }

$lastOutput = $null
foreach ($logPath in @($stdout, $stderr)) {
  if ($logPath -and (Test-Path -Path $logPath -PathType Leaf)) {
    $item = Get-Item $logPath
    if ($null -eq $lastOutput -or $item.LastWriteTime -gt $lastOutput) { $lastOutput = $item.LastWriteTime }
  }
}

$done = 0
$predictionRows = 0
$sequences = 0
$missing = 0
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  $progressLine = Get-Content $stdout | Where-Object { $_ -like '*export_yolo_predictions_route_b_progress*' } | Select-Object -Last 1
  if ($progressLine) {
    try {
      $progress = $progressLine | ConvertFrom-Json
      $done = [int]$progress.images_seen
      $predictionRows = [int]$progress.prediction_rows
      $sequences = [int]$progress.sequences
      $missing = [int]$progress.missing_prediction_files
    } catch {}
  }
}

$summaryPath = if ($outRunRoot) { Join-Path $outRunRoot 'route_b_yolo_prediction_export_summary.json' } else { '' }
$lastUnit = ''
if ($summaryPath -and (Test-Path -Path $summaryPath -PathType Leaf)) {
  try {
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    $done = [int]$summary.images_seen
    $predictionRows = [int]$summary.prediction_rows
    $sequences = [int]$summary.sequences
    $missing = [int]$summary.missing_prediction_files
    $lastUnit = 'route_b_yolo_prediction_export_summary.json'
    if ($TotalImages -le 0) { $TotalImages = $done }
  } catch {}
}

$status = if ($null -ne $proc) { 'RUNNING' } else { 'NOT RUNNING' }
Write-Host ("status={0}" -f $status)
Write-Host ("done/total={0}/{1}" -f $done, $TotalImages)
Write-Host ("pid={0}" -f $pidText)
Write-Host ("start_time={0}" -f $(if ($meta.ContainsKey('started')) { $meta['started'] } else { '' }))
Write-Host ("last_output_timestamp={0}" -f $lastOutput)
Write-Host ("last_completed_unit={0}" -f $lastUnit)
Write-Host ("prediction_rows={0}" -f $predictionRows)
Write-Host ("sequences={0}" -f $sequences)
Write-Host ("missing_prediction_files={0}" -f $missing)
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
Write-Host ("out_run_root={0}" -f $outRunRoot)
if ($summaryPath -and (Test-Path -Path $summaryPath -PathType Leaf)) {
  Write-Host 'summary_head:'
  Get-Content $summaryPath -TotalCount 60
}
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host 'stdout_tail:'
  Get-Content $stdout -Tail 25
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host 'stderr_tail:'
  Get-Content $stderr -Tail 25
}

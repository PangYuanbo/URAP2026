param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\yolomg_action\gt_csv_export_detached',
  [string]$RunId = 'yolomg_export_gt_csv',
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
$outCsv = if ($meta.ContainsKey('out_csv')) { [string]$meta['out_csv'] } else { '' }
$summaryPath = if ($outCsv) { "$outCsv.summary.json" } else { '' }

$lastOutput = $null
foreach ($logPath in @($stdout, $stderr, $outCsv, $summaryPath)) {
  if ($logPath -and (Test-Path -Path $logPath -PathType Leaf)) {
    $item = Get-Item $logPath
    if ($null -eq $lastOutput -or $item.LastWriteTime -gt $lastOutput) { $lastOutput = $item.LastWriteTime }
  }
}

$done = 0
$labels = 0
$sequences = 0
$lastUnit = ''
if ($summaryPath -and (Test-Path -Path $summaryPath -PathType Leaf)) {
  try {
    $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
    $done = [int]$summary.images_seen
    $labels = [int]$summary.labels_seen
    $sequences = [int]$summary.sequences
    $lastUnit = 'gt_csv_summary.json'
    if ($TotalImages -le 0) { $TotalImages = $done }
  } catch {}
} elseif ($outCsv -and (Test-Path -Path $outCsv -PathType Leaf)) {
  $lastUnit = Split-Path -Leaf $outCsv
}

$csvBytes = 0
if ($outCsv -and (Test-Path -Path $outCsv -PathType Leaf)) {
  $csvBytes = (Get-Item $outCsv).Length
}

$status = if ($null -ne $proc) { 'RUNNING' } else { 'NOT RUNNING' }
Write-Host ("status={0}" -f $status)
Write-Host ("done/total={0}/{1}" -f $done, $TotalImages)
Write-Host ("pid={0}" -f $pidText)
Write-Host ("start_time={0}" -f $(if ($meta.ContainsKey('started')) { $meta['started'] } else { '' }))
Write-Host ("last_output_timestamp={0}" -f $lastOutput)
Write-Host ("last_completed_unit={0}" -f $lastUnit)
Write-Host ("labels_seen={0}" -f $labels)
Write-Host ("sequences={0}" -f $sequences)
Write-Host ("csv_bytes={0}" -f $csvBytes)
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
Write-Host ("out_csv={0}" -f $outCsv)
if ($summaryPath -and (Test-Path -Path $summaryPath -PathType Leaf)) {
  Write-Host 'summary_head:'
  Get-Content $summaryPath -TotalCount 60
}
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host 'stdout_tail:'
  Get-Content $stdout -Tail 20
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host 'stderr_tail:'
  Get-Content $stderr -Tail 20
}

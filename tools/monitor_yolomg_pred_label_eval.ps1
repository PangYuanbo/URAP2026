param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\yolomg_pred_label_eval_runner',
  [string]$RunId = 'yolomg_pred_label_eval',
  [int]$TotalFrames = 71608
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
$outDir = if ($meta.ContainsKey('out_dir')) { [string]$meta['out_dir'] } else { '' }
$lastOutput = $null
foreach ($logPath in @($stdout, $stderr)) {
  if ($logPath -and (Test-Path -Path $logPath -PathType Leaf)) {
    $item = Get-Item $logPath
    if ($null -eq $lastOutput -or $item.LastWriteTime -gt $lastOutput) { $lastOutput = $item.LastWriteTime }
  }
}

$done = 0
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  $progressLine = Get-Content $stdout | Where-Object { $_ -like '*yolomg_eval_pred_labels_progress*' } | Select-Object -Last 1
  if ($progressLine) {
    try {
      $progress = $progressLine | ConvertFrom-Json
      $done = [int]$progress.images_done
      $TotalFrames = [int]$progress.images_total
    } catch {}
  }
}
$manifestPath = if ($outDir) { Join-Path $outDir 'manifest.json' } else { '' }
$summaryText = ''
if ($manifestPath -and (Test-Path -Path $manifestPath -PathType Leaf)) {
  try {
    $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
    $done = [int]$manifest.summary.frames
    $summaryText = ('weighted_ap50={0}; weighted_f1={1}; weighted_recall={2}; weighted_precision={3}' -f $manifest.summary.weighted_ap50, $manifest.summary.weighted_f1, $manifest.summary.weighted_recall, $manifest.summary.weighted_precision)
  } catch {}
}

$status = if ($null -ne $proc) { 'RUNNING' } else { 'NOT RUNNING' }
Write-Host ("status={0}" -f $status)
Write-Host ("done/total={0}/{1}" -f $done, $TotalFrames)
Write-Host ("pid={0}" -f $pidText)
Write-Host ("start_time={0}" -f $(if ($meta.ContainsKey('started')) { $meta['started'] } else { '' }))
Write-Host ("last_output_timestamp={0}" -f $lastOutput)
Write-Host ("last_completed_unit={0}" -f $(if ($manifestPath -and (Test-Path -Path $manifestPath -PathType Leaf)) { 'manifest.json' } else { '' }))
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
Write-Host ("out_dir={0}" -f $outDir)
if ($summaryText) { Write-Host ("summary={0}" -f $summaryText) }
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host 'stdout_tail:'
  Get-Content $stdout -Tail 25
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host 'stderr_tail:'
  Get-Content $stderr -Tail 25
}

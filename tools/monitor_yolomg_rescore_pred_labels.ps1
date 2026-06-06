param(
  [string]$RunId = 'yolomg_rescore_pred_labels',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\yolomg_action\rescore_detached'),
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

$outLabelDir = if ($meta.ContainsKey('out_label_dir')) { [string]$meta['out_label_dir'] } else { '' }
$stdout = if ($meta.ContainsKey('stdout')) { [string]$meta['stdout'] } else { '' }
$stderr = if ($meta.ContainsKey('stderr')) { [string]$meta['stderr'] } else { '' }
$summary = if ($outLabelDir) { Join-Path (Split-Path -Parent $outLabelDir) ((Split-Path -Leaf $outLabelDir) + '_rescore_summary.json') } else { '' }

$done = 0
$lastCompleted = ''
if ($outLabelDir -and (Test-Path -Path $outLabelDir -PathType Container)) {
  $files = Get-ChildItem -Path $outLabelDir -Filter '*.txt' -File -ErrorAction SilentlyContinue
  $done = @($files).Count
  $last = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($last) { $lastCompleted = $last.Name }
}
if ($summary -and (Test-Path -Path $summary -PathType Leaf)) {
  $lastCompleted = Split-Path -Leaf $summary
}

$lastOutput = $null
foreach ($path in @($stdout, $stderr, $summary)) {
  if ($path -and (Test-Path -Path $path -PathType Leaf)) {
    $t = (Get-Item $path).LastWriteTime
    if ($null -eq $lastOutput -or $t -gt $lastOutput) { $lastOutput = $t }
  }
}

$status = if ($proc -and $proc.CommandLine -like '*yolomg_rescore_pred_labels_from_tracklets.py*') { 'RUNNING' } else { 'NOT RUNNING' }
Write-Host ("status={0}" -f $status)
Write-Host ("done/total={0}/{1}" -f $done, $TotalFrames)
Write-Host ("pid={0}" -f $pidText)
Write-Host ("start_time={0}" -f $(if ($meta.ContainsKey('started')) { $meta['started'] } else { '' }))
Write-Host ("last_output_timestamp={0}" -f $lastOutput)
Write-Host ("last_completed_unit={0}" -f $lastCompleted)
Write-Host ("out_label_dir={0}" -f $outLabelDir)
Write-Host ("summary={0}" -f $summary)
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
if ($summary -and (Test-Path -Path $summary -PathType Leaf)) {
  Write-Host 'summary_tail:'
  Get-Content -Path $summary -Tail 40
}
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host 'stdout_tail:'
  Get-Content -Path $stdout -Tail 20
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host 'stderr_tail:'
  Get-Content -Path $stderr -Tail 20
}

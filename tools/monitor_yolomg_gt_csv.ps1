param(
  [string]$RunId = 'yolomg_gt_csv',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\yolomg_action\gt_csv_detached'),
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
$out = if ($meta.ContainsKey('out')) { [string]$meta['out'] } else { '' }
$stdout = if ($meta.ContainsKey('stdout')) { [string]$meta['stdout'] } else { '' }
$stderr = if ($meta.ContainsKey('stderr')) { [string]$meta['stderr'] } else { '' }

$done = 0
$labelsSeen = 0
$sequences = 0
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  $jsonLine = Get-Content $stdout | Where-Object { $_ -like '*"images_seen"*' -and $_ -like '*"labels_seen"*' } | Select-Object -Last 1
  if ($jsonLine) {
    try {
      $summary = $jsonLine | ConvertFrom-Json
      $done = [int]$summary.images_seen
      $labelsSeen = [int]$summary.labels_seen
      $sequences = [int]$summary.sequences
    } catch {}
  }
}
if ($out -and (Test-Path -Path $out -PathType Leaf)) {
  $lineCount = (Get-Content -Path $out | Measure-Object -Line).Lines
  if ($lineCount -gt 0) {
    $labelsSeen = $lineCount - 1
  }
}

$lastOutput = $null
foreach ($path in @($stdout, $stderr, $out)) {
  if ($path -and (Test-Path -Path $path -PathType Leaf)) {
    $t = (Get-Item $path).LastWriteTime
    if ($null -eq $lastOutput -or $t -gt $lastOutput) { $lastOutput = $t }
  }
}

$status = if ($proc -and $proc.CommandLine -like '*export-yolo-labels-gt-csv*') { 'RUNNING' } else { 'NOT RUNNING' }
Write-Host ("status={0}" -f $status)
Write-Host ("done/total={0}/{1}" -f $done, $TotalImages)
Write-Host ("pid={0}" -f $pidText)
Write-Host ("start_time={0}" -f $(if ($meta.ContainsKey('started')) { $meta['started'] } else { '' }))
Write-Host ("last_output_timestamp={0}" -f $lastOutput)
Write-Host ("last_completed_unit={0}" -f $(if ($out -and (Test-Path -Path $out -PathType Leaf)) { Split-Path -Leaf $out } else { '' }))
Write-Host ("labels_seen={0}" -f $labelsSeen)
Write-Host ("sequences={0}" -f $sequences)
Write-Host ("out={0}" -f $out)
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host 'stdout_tail:'
  Get-Content $stdout -Tail 30
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host 'stderr_tail:'
  Get-Content $stderr -Tail 30
}

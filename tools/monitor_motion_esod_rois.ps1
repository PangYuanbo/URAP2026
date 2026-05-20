param(
  [ValidateSet('nps','ard100')]
  [string]$Dataset = 'ard100',
  [string]$Out = 'artifacts\motion_esod_rois'
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outAbs = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $repo $Out }
$pidFile = Join-Path $outAbs "motion_esod_${Dataset}_pid.txt"
$metaFile = Join-Path $outAbs "motion_esod_${Dataset}_meta.json"
$summaryFile = Join-Path $outAbs "${Dataset}_summary.json"
$allList = Join-Path $outAbs "${Dataset}_all.txt"
$posList = Join-Path $outAbs "${Dataset}_positive.txt"

if (-not (Test-Path $pidFile)) {
  Write-Host 'NOT RUNNING'
  exit 0
}

$procId = [int](Get-Content $pidFile | Select-Object -First 1)
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $procId" -ErrorAction SilentlyContinue
$meta = if (Test-Path $metaFile) { Get-Content $metaFile -Raw | ConvertFrom-Json } else { $null }

if (-not $proc) {
  Write-Host 'NOT RUNNING'
} else {
  Write-Host 'RUNNING'
}

$allCount = if (Test-Path $allList) { (Get-Content $allList | Where-Object { $_ }).Count } else { 0 }
$posCount = if (Test-Path $posList) { (Get-Content $posList | Where-Object { $_ }).Count } else { 0 }
if (Test-Path $summaryFile) {
  $summary = Get-Content $summaryFile -Raw | ConvertFrom-Json
  $patches = ($summary.jobs | Measure-Object -Property patches_saved -Sum).Sum
  $frames = ($summary.jobs | Measure-Object -Property frames_seen -Sum).Sum
  Write-Host ("done/total: {0}/complete" -f $patches)
  Write-Host ("last_completed_unit: summary frames_seen={0} patches={1} positive={2}" -f $frames,$allCount,$posCount)
} else {
  Write-Host ("done/total: {0}/unknown" -f $allCount)
  Write-Host ("last_completed_unit: patches={0} positive={1}" -f $allCount,$posCount)
}
Write-Host ("pid: {0}" -f $procId)
if ($meta) {
  Write-Host ("start_time: {0}" -f $meta.start_time)
  if (Test-Path $meta.stdout_log) {
    Write-Host ("last_output: {0}" -f ((Get-Item $meta.stdout_log).LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')))
  }
  Write-Host ("stdout: {0}" -f $meta.stdout_log)
  Write-Host ("stderr: {0}" -f $meta.stderr_log)
  Write-Host ("output_root: {0}" -f $meta.output_root)
}
if (Test-Path $summaryFile) { Write-Host ("summary: {0}" -f $summaryFile) }
if (Test-Path $allList) { Write-Host ("all_list: {0}" -f $allList) }
if (Test-Path $posList) { Write-Host ("positive_list: {0}" -f $posList) }

param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\detached_timeline_eval',
  [string]$RunId = 'yolomg_timeline_eval',
  [string]$OutputDir = '',
  [int]$TotalFrames = 71608
)

$ErrorActionPreference = 'Stop'

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (-not (Test-Path -Path $pidFile -PathType Leaf)) {
  Write-Host "NOT RUNNING: pid file missing: $pidFile"
  exit 0
}

$pidText = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
$proc = $null
if ($pidText -match '^\d+$') {
  $proc = Get-Process -Id ([int]$pidText) -ErrorAction SilentlyContinue
}

$meta = @{}
if (Test-Path -Path $metaFile -PathType Leaf) {
  foreach ($line in Get-Content $metaFile) {
    $parts = $line -split '=', 2
    if ($parts.Count -eq 2) {
      $meta[$parts[0]] = $parts[1]
    }
  }
}

if (-not $OutputDir -and $meta.ContainsKey('output_dir')) {
  $OutputDir = [string]$meta['output_dir']
}

$stdout = if ($meta.ContainsKey('stdout')) { [string]$meta['stdout'] } else { '' }
$stderr = if ($meta.ContainsKey('stderr')) { [string]$meta['stderr'] } else { '' }
$lastOutput = $null
foreach ($logPath in @($stdout, $stderr)) {
  if ($logPath -and (Test-Path -Path $logPath -PathType Leaf)) {
    $item = Get-Item $logPath
    if ($null -eq $lastOutput -or $item.LastWriteTime -gt $lastOutput) {
      $lastOutput = $item.LastWriteTime
    }
  }
}

$done = 0
$predCount = 0
$lastCompleted = ''
if ($OutputDir -and (Test-Path -Path $OutputDir -PathType Container)) {
  $perFrameDir = Join-Path $OutputDir 'per_frame'
  if (Test-Path -Path $perFrameDir -PathType Container) {
    $csvs = Get-ChildItem -Path $perFrameDir -Filter '*_per_frame.csv' -File -ErrorAction SilentlyContinue
    foreach ($csv in $csvs) {
      $lineCount = 0
      Get-Content $csv.FullName | ForEach-Object { $lineCount += 1 }
      if ($lineCount -gt 0) {
        $done += ($lineCount - 1)
      }
    }
    $lastCsv = $csvs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($null -ne $lastCsv) {
      $lastCompleted = $lastCsv.Name
    }
  }
  $predLabelsDir = Join-Path $OutputDir 'pred_labels'
  if (Test-Path -Path $predLabelsDir -PathType Container) {
    $predCount = @(Get-ChildItem -Path $predLabelsDir -Filter '*.txt' -File -ErrorAction SilentlyContinue).Count
    if ($done -eq 0) {
      $done = $predCount
    }
  }
}

$gpuText = 'nvidia-smi unavailable'
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($null -ne $smi) {
  $gpuText = (& nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null) -join '; '
}

$status = if ($null -ne $proc) { 'RUNNING' } else { 'NOT RUNNING' }
Write-Host ("status={0}" -f $status)
Write-Host ("done/total={0}/{1}" -f $done, $TotalFrames)
Write-Host ("pid={0}" -f $pidText)
Write-Host ("start_time={0}" -f $(if ($meta.ContainsKey('started')) { $meta['started'] } elseif ($null -ne $proc) { $proc.StartTime } else { '' }))
Write-Host ("last_output_timestamp={0}" -f $lastOutput)
Write-Host ("last_completed_unit={0}" -f $lastCompleted)
Write-Host ("pred_label_files={0}" -f $predCount)
Write-Host ("stdout={0}" -f $stdout)
Write-Host ("stderr={0}" -f $stderr)
Write-Host ("output_dir={0}" -f $OutputDir)
Write-Host ("gpu={0}" -f $gpuText)

if ($stdout -and (Test-Path -Path $stdout -PathType Leaf)) {
  Write-Host 'stdout_tail:'
  Get-Content $stdout -Tail 20
}
if ($stderr -and (Test-Path -Path $stderr -PathType Leaf)) {
  Write-Host 'stderr_tail:'
  Get-Content $stderr -Tail 20
}

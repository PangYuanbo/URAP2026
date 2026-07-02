param(
  [string]$Out = 'runs\window_accuracy\detached_yolo_eval',
  [string]$RunId = 'paper_window_accuracy',
  [int]$TailLines = 80
)

$ErrorActionPreference = 'Stop'

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outAbs = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $repo $Out }
$pidFile = Join-Path $outAbs ("{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $outAbs ("{0}_meta.json" -f $RunId)
$summaryFile = Join-Path $outAbs 'summary.json'
$metricsFile = Join-Path $outAbs 'per_frame_window_metrics.csv'
$worstFile = Join-Path $outAbs 'worst_windows.csv'

if (-not (Test-Path -Path $metaFile -PathType Leaf)) {
  Write-Host 'NOT RUNNING'
  Write-Host ("meta_not_found: {0}" -f $metaFile)
  exit 0
}

$meta = Get-Content $metaFile -Raw | ConvertFrom-Json
$pidValue = $null
if (Test-Path -Path $pidFile -PathType Leaf) {
  $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$proc = $null
if ($pidValue -match '^\d+$') {
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
}

if ($proc) {
  Write-Host 'RUNNING'
} else {
  Write-Host 'NOT RUNNING'
}

$labelsDir = [string]$meta.prediction_labels_dir
$labelCount = 0
$latestLabel = $null
if ($labelsDir -and (Test-Path -Path $labelsDir -PathType Container)) {
  $labels = Get-ChildItem -Path $labelsDir -Recurse -File -Filter '*.txt' -ErrorAction SilentlyContinue
  $labelCount = @($labels).Count
  $latestLabel = $labels | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

$done = 0
$lastUnit = 'launched'
if (Test-Path -Path $summaryFile -PathType Leaf) {
  $done = 2
  $summary = Get-Content $summaryFile -Raw | ConvertFrom-Json
  $lastUnit = "curves_complete videos=$($summary.videos) frames=$($summary.frames)"
} elseif ($labelCount -gt 0) {
  $done = 1
  $lastUnit = "prediction_labels=$labelCount"
} elseif ($meta.stdout_log -and (Test-Path -Path $meta.stdout_log -PathType Leaf)) {
  $tail = Get-Content -Path $meta.stdout_log -Tail 20 -ErrorAction SilentlyContinue
  if ($tail) { $lastUnit = ($tail | Select-Object -Last 1) }
}

Write-Host ("done/total: {0}/2" -f $done)
Write-Host ("pid: {0}" -f $pidValue)
if ($proc) {
  $procStart = $proc.CreationDate
  if ($procStart -is [string]) {
    $procStart = [Management.ManagementDateTimeConverter]::ToDateTime($procStart)
  }
  Write-Host ("pid_start: {0}" -f $procStart.ToString('yyyy-MM-dd HH:mm:ss'))
  Write-Host ("pid_command_line: {0}" -f $proc.CommandLine)
}
Write-Host ("start_time: {0}" -f $meta.start_time)
Write-Host ("last_completed_unit: {0}" -f $lastUnit)
Write-Host ("output_root: {0}" -f $outAbs)
Write-Host ("prediction_labels_dir: {0}" -f $labelsDir)
Write-Host ("prediction_label_count: {0}" -f $labelCount)

$lastOutputs = @()
foreach ($path in @($meta.stdout_log, $meta.stderr_log, $summaryFile, $metricsFile, $worstFile)) {
  if ($path -and (Test-Path -Path $path -PathType Leaf)) { $lastOutputs += (Get-Item $path) }
}
if ($latestLabel) { $lastOutputs += $latestLabel }
$latestOutput = $lastOutputs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latestOutput) {
  Write-Host ("last_output_timestamp: {0}" -f $latestOutput.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  Write-Host ("last_output_path: {0}" -f $latestOutput.FullName)
} else {
  Write-Host 'last_output_timestamp: none'
}

Write-Host ("stdout: {0}" -f $meta.stdout_log)
Write-Host ("stderr: {0}" -f $meta.stderr_log)
if (Test-Path -Path $summaryFile -PathType Leaf) { Write-Host ("summary: {0}" -f $summaryFile) }
if (Test-Path -Path $metricsFile -PathType Leaf) { Write-Host ("per_frame_csv: {0}" -f $metricsFile) }
if (Test-Path -Path $worstFile -PathType Leaf) { Write-Host ("worst_windows_csv: {0}" -f $worstFile) }

if ($meta.stdout_log -and (Test-Path -Path $meta.stdout_log -PathType Leaf)) {
  Write-Host ''
  Write-Host '== stdout tail =='
  Get-Content -Path $meta.stdout_log -Tail $TailLines -ErrorAction SilentlyContinue
}
if ($meta.stderr_log -and (Test-Path -Path $meta.stderr_log -PathType Leaf)) {
  Write-Host ''
  Write-Host '== stderr tail =='
  Get-Content -Path $meta.stderr_log -Tail $TailLines -ErrorAction SilentlyContinue
}

$gpuLine = (& nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
if ($LASTEXITCODE -eq 0 -and $gpuLine) {
  Write-Host ''
  Write-Host ("gpu_signal: {0}" -f $gpuLine)
}

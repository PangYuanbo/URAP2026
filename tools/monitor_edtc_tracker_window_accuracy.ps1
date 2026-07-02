param(
  [string]$Out = 'runs\window_accuracy\papers\edtc_antiuav600',
  [string]$RunId = 'edtc_antiuav600',
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
$segmentsFile = Join-Path $outAbs 'low_accuracy_segments.csv'

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

$resultCount = 0
$latestResult = $null
$resultsDir = [string]$meta.results_dir
if ($resultsDir -and (Test-Path -Path $resultsDir -PathType Container)) {
  $resultFiles = Get-ChildItem -Path $resultsDir -File -Filter '*.txt' -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notlike '*_time.txt' -and $_.Name -notlike '*_all_*' }
  $resultCount = @($resultFiles).Count
  $latestResult = $resultFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
}

$done = $resultCount
$total = [int]($meta.total_sequences)
$lastUnit = if ($latestResult) { "tracker_result=$($latestResult.Name)" } else { 'launched' }
if (Test-Path -Path $summaryFile -PathType Leaf) {
  $summary = Get-Content $summaryFile -Raw | ConvertFrom-Json
  $done = $total
  $lastUnit = "curves_complete videos=$($summary.videos) frames=$($summary.frames)"
}

Write-Host ("done/total: {0}/{1}" -f $done, $total)
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
Write-Host ("results_dir: {0}" -f $resultsDir)

$lastOutputs = @()
foreach ($path in @($meta.stdout_log, $meta.stderr_log, $summaryFile, $metricsFile, $worstFile, $segmentsFile)) {
  if ($path -and (Test-Path -Path $path -PathType Leaf)) { $lastOutputs += (Get-Item $path) }
}
if ($latestResult) { $lastOutputs += $latestResult }
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
if (Test-Path -Path $segmentsFile -PathType Leaf) { Write-Host ("low_accuracy_segments_csv: {0}" -f $segmentsFile) }

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

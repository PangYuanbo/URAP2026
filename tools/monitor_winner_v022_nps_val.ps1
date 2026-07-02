param(
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val",
  [string]$RunId = "nps_val",
  [string]$JobId = "winner_v022_nps_val",
  [int]$TailLines = 80
)

$ErrorActionPreference = "Stop"

$runOut = Join-Path $OutputRoot $RunId
$pidFile = Join-Path $runOut ("{0}_pid.txt" -f $JobId)
$metaFile = Join-Path $runOut ("{0}_meta.json" -f $JobId)

if (-not (Test-Path -Path $metaFile -PathType Leaf)) {
  Write-Host "NOT RUNNING"
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
  Write-Host "RUNNING"
} else {
  Write-Host "NOT RUNNING"
}

$dirs = @()
if (Test-Path -Path $runOut -PathType Container) {
  $dirs = @(Get-ChildItem -Path $runOut -Directory -ErrorAction SilentlyContinue)
}
$done = @($dirs | Where-Object { Test-Path (Join-Path $_.FullName "result.json") })
$total = [int]($meta.total_clips)
if ($total -le 0) { $total = $dirs.Count }

$latestResult = $done | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$lastUnit = if ($latestResult) { "last_result=$($latestResult.Name)" } else { "launched" }

Write-Host ("done/total: {0}/{1}" -f $done.Count, $total)
Write-Host ("pid: {0}" -f $pidValue)
if ($proc) {
  $procStart = $proc.CreationDate
  if ($procStart -is [string]) {
    $procStart = [Management.ManagementDateTimeConverter]::ToDateTime($procStart)
  }
  Write-Host ("pid_start: {0}" -f $procStart.ToString("yyyy-MM-dd HH:mm:ss"))
  Write-Host ("pid_command_line: {0}" -f $proc.CommandLine)
}
Write-Host ("start_time: {0}" -f $meta.start_time)
Write-Host ("last_completed_unit: {0}" -f $lastUnit)
Write-Host ("output_root: {0}" -f $OutputRoot)
Write-Host ("run_output: {0}" -f $runOut)

$lastOutputs = @()
foreach ($path in @($meta.stdout_log, $meta.stderr_log)) {
  if ($path -and (Test-Path -Path $path -PathType Leaf)) { $lastOutputs += (Get-Item $path) }
}
if ($latestResult) {
  $resultFile = Join-Path $latestResult.FullName "result.json"
  if (Test-Path -Path $resultFile -PathType Leaf) { $lastOutputs += (Get-Item $resultFile) }
}
$latestOutput = $lastOutputs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latestOutput) {
  Write-Host ("last_output_timestamp: {0}" -f $latestOutput.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"))
  Write-Host ("last_output_path: {0}" -f $latestOutput.FullName)
} else {
  Write-Host "last_output_timestamp: none"
}

Write-Host ("stdout: {0}" -f $meta.stdout_log)
Write-Host ("stderr: {0}" -f $meta.stderr_log)

if ($meta.stdout_log -and (Test-Path -Path $meta.stdout_log -PathType Leaf)) {
  Write-Host ""
  Write-Host "== stdout tail =="
  Get-Content -Path $meta.stdout_log -Tail $TailLines -ErrorAction SilentlyContinue
}
if ($meta.stderr_log -and (Test-Path -Path $meta.stderr_log -PathType Leaf)) {
  Write-Host ""
  Write-Host "== stderr tail =="
  Get-Content -Path $meta.stderr_log -Tail $TailLines -ErrorAction SilentlyContinue
}

$gpuLine = (& nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
if ($LASTEXITCODE -eq 0 -and $gpuLine) {
  Write-Host ""
  Write-Host ("gpu_signal: {0}" -f $gpuLine)
}

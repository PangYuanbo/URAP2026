param(
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\ego_adaptive_vatd\nps_posttrain_watcher',
  [string]$RunId = 'ego_adaptive_nps_posttrain_watcher',
  [int]$TailLines = 40,
  [int]$StaleSeconds = 600
)

$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

Write-Host '== Meta =='
if (Test-Path $metaFile) {
  Get-Content $metaFile
} else {
  Write-Host "meta missing: $metaFile"
}

$pidValue = $null
if (Test-Path $pidFile) {
  $pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
}
$process = $null
if ($pidValue) {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
}
if ($process -and $process.CommandLine -like "*$RunId.runner.ps1*") {
  Write-Host ""
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else {
  Write-Host ""
  Write-Host "NOT RUNNING PID=$pidValue"
}

$meta = @{}
if (Test-Path $metaFile) {
  foreach ($line in Get-Content $metaFile) {
    $idx = $line.IndexOf('=')
    if ($idx -gt 0) { $meta[$line.Substring(0, $idx)] = $line.Substring($idx + 1) }
  }
}

$stdout = $meta['stdout']
$stderr = $meta['stderr']
$weights = $meta['weights']
$comparisonJson = $meta['comparison_json']
$target20Json = $meta['target20_json']
$scoreOutputRoot = $meta['score_output_root']
$scoreRunId = $meta['score_run_id']

$done = 0
$total = 3
$lastUnit = 'wait_weights'
if ($weights -and (Test-Path $weights)) {
  $done = 1
  $lastUnit = 'weights_ready'
}
if ($comparisonJson -and (Test-Path $comparisonJson)) {
  $done = 2
  $lastUnit = 'comparison_ready'
}
if ($target20Json -and (Test-Path $target20Json)) {
  $done = 3
  $lastUnit = 'target20_gate'
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $weights, $comparisonJson, $target20Json)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

Write-Host ""
Write-Host "done/total: $done/$total"
Write-Host "last output timestamp: $lastWrite"
if ($lastWrite) {
  $ageSeconds = [int]((Get-Date) - $lastWrite).TotalSeconds
  Write-Host "progress_age_seconds: $ageSeconds"
  Write-Host ("progress_stale: {0}" -f ($ageSeconds -gt $StaleSeconds))
  Write-Host "progress_stale_threshold_seconds: $StaleSeconds"
} else {
  Write-Host "progress_age_seconds: missing"
  Write-Host "progress_stale: unknown"
  Write-Host "progress_stale_threshold_seconds: $StaleSeconds"
}
Write-Host "last completed unit: $lastUnit"
Write-Host "weights: $weights"
Write-Host "comparison_json: $comparisonJson"
Write-Host "target20_json: $target20Json"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

if ($target20Json -and (Test-Path $target20Json)) {
  Write-Host ""
  Write-Host "== 20pct gate =="
  try { Get-Content $target20Json -Raw | ConvertFrom-Json | Select-Object status,reason,wins,primary_metric,direction,baseline_value,min_relative_improvement | ConvertTo-Json -Depth 6 } catch {}
}

Write-Host ""
Write-Host "== Score sweep =="
if ($scoreOutputRoot -and (Test-Path $scoreOutputRoot)) {
  & (Join-Path $PSScriptRoot 'monitor_nps_ego_adaptive_vatd_score_sweep.ps1') -OutputRoot $scoreOutputRoot -RunId $scoreRunId -TailLines 20
} else {
  Write-Host "score_output_root_missing=$scoreOutputRoot"
}

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }

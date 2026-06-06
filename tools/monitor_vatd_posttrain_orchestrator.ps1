param(
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\vatd_posttrain_orchestrator_20260605'),
  [string]$RunId = 'vatd_posttrain_orchestrator_20260605',
  [int]$TailLines = 40
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
$aotWeights = $meta['aot_weights']
$npsWeights = $meta['nps_weights']
$aotScoreMarker = Join-Path $OutputRoot 'aot_score_launched.marker'
$aotEvalMarker = Join-Path $OutputRoot 'aot_official_launched.marker'
$npsMarker = Join-Path $OutputRoot 'nps_score_sweep_launched.marker'

$done = 0
$total = 3
$lastUnit = 'waiting'
if (Test-Path $aotScoreMarker) { $done += 1; $lastUnit = 'aot_score_launched' }
if (Test-Path $aotEvalMarker) { $done += 1; $lastUnit = 'aot_official_launched' }
if (Test-Path $npsMarker) { $done += 1; $lastUnit = 'nps_score_sweep_launched' }

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $aotScoreMarker, $aotEvalMarker, $npsMarker)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

Write-Host ""
Write-Host "done/total: $done/$total"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: $lastUnit"
Write-Host "aot_weights_ready=$(if ($aotWeights -and (Test-Path $aotWeights)) { 'true' } else { 'false' })"
Write-Host "nps_weights_ready=$(if ($npsWeights -and (Test-Path $npsWeights)) { 'true' } else { 'false' })"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }

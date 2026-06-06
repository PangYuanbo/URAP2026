param(
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\nps_sota_research\nps_ego_adaptive_vatd_score_sweep_runner'),
  [string]$RunId = 'nps_ego_adaptive_vatd_score_sweep',
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

if ($process -and $process.CommandLine -like "*$RunId*") {
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
$scoreOut = $meta['score_out']
$attachedOut = $meta['attached_out']
$sweepOut = $meta['sweep_out_json']
$comparisonJson = $meta['comparison_out_json']
if (-not $comparisonJson -and $sweepOut) {
  $comparisonJson = $sweepOut -replace '\.json$', '_comparison.json'
}
$claimGateJson = $null
if ($comparisonJson) {
  $comparisonPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $comparisonJson))
  $claimGateJson = [System.IO.Path]::Combine(
    [System.IO.Path]::GetDirectoryName($comparisonPath),
    ([System.IO.Path]::GetFileNameWithoutExtension($comparisonPath) + '_claim_gate.json')
  )
}

$done = 0
$total = 4
$lastUnit = 'not started'
if ($scoreOut -and (Test-Path $scoreOut)) {
  $done = 1
  $lastUnit = 'score'
}
if ($attachedOut -and (Test-Path $attachedOut)) {
  $done = 2
  $lastUnit = 'attach'
}
if ($sweepOut -and (Test-Path $sweepOut)) {
  $done = 3
  $lastUnit = 'sweep'
}
if ($claimGateJson -and (Test-Path $claimGateJson)) {
  $done = 4
  $lastUnit = 'claim_gate'
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $scoreOut, $attachedOut, $sweepOut, $comparisonJson, $claimGateJson)) {
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
Write-Host "score jsonl: $scoreOut"
Write-Host "attached jsonl: $attachedOut"
Write-Host "sweep json: $sweepOut"
Write-Host "comparison json: $comparisonJson"
Write-Host "claim gate json: $claimGateJson"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

Write-Host ""
Write-Host "== GPU signal =="
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
  & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
  Write-Host "-- GPU process sample --"
  & nvidia-smi pmon -c 1 | Select-String -Pattern '(^#|python)'
} else {
  Write-Host "nvidia-smi not found"
}

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }

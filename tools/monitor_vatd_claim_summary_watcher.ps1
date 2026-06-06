param(
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\vatd_claim_summary_final_runner_20260605'),
  [string]$RunId = 'vatd_claim_summary_final_20260605',
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

Write-Host ''
if ($process -and $process.CommandLine -like "*$RunId.runner.ps1*") {
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else {
  Write-Host "NOT RUNNING PID=$pidValue"
}

$meta = @{}
if (Test-Path $metaFile) {
  foreach ($line in Get-Content $metaFile) {
    $idx = $line.IndexOf('=')
    if ($idx -gt 0) { $meta[$line.Substring(0, $idx)] = $line.Substring($idx + 1) }
  }
}

$aotGate = $meta['aot_gate']
$npsGate = $meta['nps_gate']
$outJson = $meta['out_json']
$outMd = $meta['out_md']
$stdout = $meta['stdout']
$stderr = $meta['stderr']

$aotReady = $aotGate -and (Test-Path $aotGate)
$npsReady = $npsGate -and (Test-Path $npsGate)
$summaryReady = $outJson -and (Test-Path $outJson)

$done = 0
$total = 3
$lastUnit = 'waiting'
if ($aotReady) {
  $done += 1
  $lastUnit = 'aot_gate'
}
if ($npsReady) {
  $done += 1
  $lastUnit = 'nps_gate'
}
if ($summaryReady) {
  $done = 3
  $lastUnit = 'claim_summary'
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $aotGate, $npsGate, $outJson, $outMd)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

Write-Host ''
Write-Host "done/total: $done/$total"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: $lastUnit"
Write-Host "aot_gate_ready: $aotReady"
Write-Host "nps_gate_ready: $npsReady"
Write-Host "summary_json: $outJson"
Write-Host "summary_md: $outMd"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

if ($summaryReady) {
  Write-Host ''
  Write-Host '== Claim summary =='
  Get-Content $outJson -Raw
}

Write-Host ''
Write-Host '== stdout tail =='
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ''
Write-Host '== stderr tail =='
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }

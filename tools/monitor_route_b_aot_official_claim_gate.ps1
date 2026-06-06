param(
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\route_b_official\aot_official_claim_gate_runner'),
  [string]$RunId = 'route_b_aot_official_claim_gate',
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

$summaryGlob = $meta['summary_glob']
$outCsv = $meta['out_csv']
$outJson = $meta['out_json']
$stdout = $meta['stdout']
$stderr = $meta['stderr']
$claimGateJson = $null
if ($outCsv) {
  $csvPath = [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $outCsv))
  $claimGateJson = [System.IO.Path]::Combine(
    [System.IO.Path]::GetDirectoryName($csvPath),
    ([System.IO.Path]::GetFileNameWithoutExtension($csvPath) + '_claim_gate.json')
  )
}

$matches = @()
if ($summaryGlob) {
  $matches = @(Get-ChildItem -Path $summaryGlob -File -ErrorAction SilentlyContinue)
}

$done = 0
$total = 2
$lastUnit = 'waiting'
if ($matches.Count -gt 0) {
  $done = 1
  $lastUnit = 'official_summary'
}
if ($claimGateJson -and (Test-Path $claimGateJson)) {
  $done = 2
  $lastUnit = 'claim_gate'
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $outCsv, $outJson, $claimGateJson)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}
foreach ($match in $matches) {
  if (-not $lastWrite -or $match.LastWriteTime -gt $lastWrite) { $lastWrite = $match.LastWriteTime }
}

Write-Host ''
Write-Host "done/total: $done/$total"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: $lastUnit"
Write-Host "summary matches: $($matches.Count)"
Write-Host "claim gate json: $claimGateJson"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

if ($claimGateJson -and (Test-Path $claimGateJson)) {
  Write-Host ''
  Write-Host '== Claim gate =='
  Get-Content $claimGateJson -Raw
}

Write-Host ''
Write-Host '== stdout tail =='
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ''
Write-Host '== stderr tail =='
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }

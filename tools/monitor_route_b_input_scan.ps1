param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_input_scan'),
  [string]$RunId = 'route_b_input_scan',
  [int]$TailLines = 80
)

$ErrorActionPreference = 'Stop'

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (-not (Test-Path -Path $metaFile -PathType Leaf)) {
  Write-Host "Meta file not found: $metaFile"
  exit 1
}

$meta = Get-Content $metaFile
Write-Host '== Meta =='
$meta | Select-Object -First 140

function Get-MetaValue([string]$Key) {
  $line = ($meta | Where-Object { $_ -like "$Key=*" } | Select-Object -First 1)
  if ($line) { return $line.Substring($Key.Length + 1) }
  return $null
}

$pidValue = $null
if (Test-Path -Path $pidFile -PathType Leaf) {
  $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$process = $null
if ($pidValue -match '^\d+$') {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
}

Write-Host ''
if ($process -and $process.CommandLine -like '*scan-route-b-proposal-inputs*') {
  Write-Host ('RUNNING=true PID={0}' -f $pidValue)
  Write-Host ('PID_START={0}' -f ([Management.ManagementDateTimeConverter]::ToDateTime($process.CreationDate).ToString('yyyy-MM-dd HH:mm:ss')))
  Write-Host ('PROCESS_COMMAND={0}' -f $process.CommandLine)
} else {
  Write-Host ('NOT RUNNING PID={0}' -f $pidValue)
}

$stdoutPath = Get-MetaValue 'stdout'
$stderrPath = Get-MetaValue 'stderr'
$outPath = Get-MetaValue 'out'
$scanRootsRaw = Get-MetaValue 'scan_roots'
$scanRootCount = if ($scanRootsRaw) { @($scanRootsRaw -split ';' | Where-Object { $_ }).Count } else { 0 }
$totalUnits = [Math]::Max(1, $scanRootCount)
$doneUnits = 0
$lastCompleted = ''
$lastOutputTimes = @()

if ($outPath -and (Test-Path -Path $outPath -PathType Leaf)) {
  $doneUnits = $totalUnits
  $lastCompleted = 'proposal_input_scan'
  $lastOutputTimes += (Get-Item $outPath).LastWriteTime
  try {
    $scan = Get-Content -Path $outPath -Raw | ConvertFrom-Json
    Write-Host ('SCAN_JSON={0}' -f $outPath)
    Write-Host ('FILES_SCANNED={0}' -f $scan.files_scanned)
    Write-Host ('RUN_CANDIDATES={0}' -f $scan.num_run_candidates)
    Write-Host ('GT_CANDIDATES={0}' -f $scan.num_gt_candidates)
    Write-Host ('SUGGESTED_INPUTS={0}' -f @($scan.suggested_manifest_inputs).Count)
    if ($scan.issues -and $scan.issues.Count -gt 0) {
      Write-Host 'SCAN_ISSUES='
      $scan.issues | Select-Object -First 20
    }
  } catch {
    Write-Host ('SCAN_PARSE_ERROR={0}' -f $_.Exception.Message)
  }
}

if ($stdoutPath -and (Test-Path -Path $stdoutPath -PathType Leaf)) { $lastOutputTimes += (Get-Item $stdoutPath).LastWriteTime }
if ($stderrPath -and (Test-Path -Path $stderrPath -PathType Leaf)) { $lastOutputTimes += (Get-Item $stderrPath).LastWriteTime }

$lastOutput = 'missing'
if ($lastOutputTimes.Count -gt 0) {
  $lastOutput = ($lastOutputTimes | Sort-Object -Descending | Select-Object -First 1).ToString('yyyy-MM-dd HH:mm:ss')
}

Write-Host ''
Write-Host ('done/total: {0}/{1}' -f $doneUnits, $totalUnits)
Write-Host ('last output timestamp: {0}' -f $lastOutput)
Write-Host ('last completed unit: {0}' -f $lastCompleted)
Write-Host ('stdout log: {0}' -f $stdoutPath)
Write-Host ('stderr log: {0}' -f $stderrPath)

if ($stdoutPath -and (Test-Path -Path $stdoutPath -PathType Leaf)) {
  Write-Host ''
  Write-Host '== stdout tail =='
  Get-Content -Path $stdoutPath -Tail $TailLines -ErrorAction SilentlyContinue
}
if ($stderrPath -and (Test-Path -Path $stderrPath -PathType Leaf)) {
  $stderrTail = Get-Content -Path $stderrPath -Tail $TailLines -ErrorAction SilentlyContinue
  if ($stderrTail) {
    Write-Host ''
    Write-Host '== stderr tail =='
    $stderrTail
  }
}

$gpuLine = (& nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
if ($LASTEXITCODE -eq 0 -and $gpuLine) {
  Write-Host ''
  Write-Host ('GPU_SIGNAL={0}' -f $gpuLine)
}

param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_proposal_benchmark'),
  [string]$RunId = 'route_b_proposal_benchmark',
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
$meta | Select-Object -First 160

$pidValue = $null
if (Test-Path -Path $pidFile -PathType Leaf) {
  $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$process = $null
if ($pidValue -match '^\d+$') {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
}

Write-Host ''
if ($process -and $process.CommandLine -like '*run-multisource-proposal-policy-benchmark*') {
  Write-Host ('RUNNING=true PID={0}' -f $pidValue)
  Write-Host ('PID_START={0}' -f ([Management.ManagementDateTimeConverter]::ToDateTime($process.CreationDate).ToString('yyyy-MM-dd HH:mm:ss')))
  Write-Host ('PROCESS_COMMAND={0}' -f $process.CommandLine)
} else {
  Write-Host ('NOT RUNNING PID={0}' -f $pidValue)
}

function Get-MetaValue([string]$Key) {
  $line = ($meta | Where-Object { $_ -like "$Key=*" } | Select-Object -First 1)
  if ($line) { return $line.Substring($Key.Length + 1) }
  return $null
}

$stdoutPath = Get-MetaValue 'stdout'
$stderrPath = Get-MetaValue 'stderr'
$preflightPath = Get-MetaValue 'preflight'
$outDir = Get-MetaValue 'out_dir'
$trainRootsRaw = Get-MetaValue 'train_run_roots'
$evalRootsRaw = Get-MetaValue 'eval_run_roots'

$trainCount = if ($trainRootsRaw) { @($trainRootsRaw -split ';' | Where-Object { $_ }).Count } else { 0 }
$evalCount = if ($evalRootsRaw) { @($evalRootsRaw -split ';' | Where-Object { $_ }).Count } else { 0 }
$totalUnits = [Math]::Max(1, $trainCount + $evalCount) + 1
$doneUnits = 0
$lastCompleted = ''
$lastOutputTimes = @()

if ($outDir -and (Test-Path -Path $outDir -PathType Container)) {
  Write-Host ('OUTPUT_DIR={0}' -f $outDir)
  $proposalRoot = Join-Path $outDir 'proposal_tracklets'
  if (Test-Path -Path $proposalRoot -PathType Container) {
    $proposalSummaries = Get-ChildItem -Path $proposalRoot -Recurse -Filter 'summary.json' -File -ErrorAction SilentlyContinue
    foreach ($summary in $proposalSummaries) {
      $doneUnits += 1
      $lastCompleted = ('proposal/{0}/{1}' -f $summary.Directory.Parent.Name, $summary.Directory.Name)
      $lastOutputTimes += $summary.LastWriteTime
    }
    Write-Host ('PROPOSAL_SUMMARIES={0}' -f $proposalSummaries.Count)
  }

  $benchmarkSummary = Join-Path $outDir 'benchmark\multisource_tracklet_policy_benchmark_summary.json'
  if (Test-Path -Path $benchmarkSummary -PathType Leaf) {
    $doneUnits += 1
    $lastCompleted = 'benchmark'
    $lastOutputTimes += (Get-Item $benchmarkSummary).LastWriteTime
  }
  $proposalBenchmarkSummary = Join-Path $outDir 'multisource_proposal_policy_benchmark_summary.json'
  if (Test-Path -Path $proposalBenchmarkSummary -PathType Leaf) {
    $lastCompleted = 'proposal_benchmark_summary'
    $lastOutputTimes += (Get-Item $proposalBenchmarkSummary).LastWriteTime
  }
  $baselineReport = Join-Path $outDir 'benchmark\baseline_report\route_b_baseline_report.md'
  if (Test-Path -Path $baselineReport -PathType Leaf) {
    Write-Host ('BASELINE_REPORT={0}' -f $baselineReport)
    $lastOutputTimes += (Get-Item $baselineReport).LastWriteTime
  }
}

if ($preflightPath -and (Test-Path -Path $preflightPath -PathType Leaf)) {
  $lastOutputTimes += (Get-Item $preflightPath).LastWriteTime
  try {
    $preflight = Get-Content -Path $preflightPath -Raw | ConvertFrom-Json
    Write-Host ('PREFLIGHT={0}' -f $preflightPath)
    Write-Host ('PREFLIGHT_VALID={0}' -f $preflight.valid)
    Write-Host ('PREFLIGHT_BBOX_ROWS train={0} eval={1}' -f $preflight.train_bbox_rows, $preflight.eval_bbox_rows)
    if ($preflight.issues -and $preflight.issues.Count -gt 0) {
      Write-Host 'PREFLIGHT_ISSUES='
      $preflight.issues | Select-Object -First 20
    }
  } catch {
    Write-Host ('PREFLIGHT_PARSE_ERROR={0}' -f $_.Exception.Message)
  }
}

if ($stdoutPath -and (Test-Path -Path $stdoutPath -PathType Leaf)) { $lastOutputTimes += (Get-Item $stdoutPath).LastWriteTime }
if ($stderrPath -and (Test-Path -Path $stderrPath -PathType Leaf)) { $lastOutputTimes += (Get-Item $stderrPath).LastWriteTime }

if ($doneUnits -gt $totalUnits) { $doneUnits = $totalUnits }
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

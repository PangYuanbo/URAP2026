param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_multisource_benchmark'),
  [string]$RunId = 'route_b_multisource_benchmark',
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

$pidValue = $null
if (Test-Path -Path $pidFile -PathType Leaf) {
  $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$process = $null
if ($pidValue -match '^\d+$') {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
}

Write-Host ''
if ($process -and $process.CommandLine -like '*run-multisource-tracklet-policy-benchmark*') {
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
$outDir = Get-MetaValue 'out_dir'
$preflightPath = Get-MetaValue 'preflight'
$evalDatasetsRaw = Get-MetaValue 'eval_datasets'
$evalInputsRaw = Get-MetaValue 'eval_inputs'

$evalNames = @()
if ($evalDatasetsRaw) {
  $evalNames = @($evalDatasetsRaw -split ';' | Where-Object { $_ })
}
if ($evalNames.Count -eq 0 -and $evalInputsRaw) {
  $evalNames = @($evalInputsRaw -split ';' | Where-Object { $_ } | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_) })
}

$totalUnits = 1 + [Math]::Max(1, $evalNames.Count)
$doneUnits = 0
$lastCompleted = ''
$lastOutputTimes = @()

if ($outDir -and (Test-Path -Path $outDir -PathType Container)) {
  Write-Host ('OUTPUT_DIR={0}' -f $outDir)
  $trainSummary = Join-Path $outDir 'train\multisource_tracklet_action_policy_experiment_summary.json'
  if (Test-Path -Path $trainSummary -PathType Leaf) {
    $doneUnits += 1
    $lastCompleted = 'train'
    $lastOutputTimes += (Get-Item $trainSummary).LastWriteTime
  }

  $evalRoot = Join-Path $outDir 'eval'
  if (Test-Path -Path $evalRoot -PathType Container) {
    $evalSummaries = Get-ChildItem -Path $evalRoot -Recurse -Filter 'action_dynamics_pipeline_summary.json' -File -ErrorAction SilentlyContinue
    foreach ($summary in $evalSummaries) {
      $doneUnits += 1
      $lastCompleted = ('eval/{0}' -f $summary.Directory.Name)
      $lastOutputTimes += $summary.LastWriteTime
    }
    Write-Host ('EVAL_SUMMARIES={0}' -f $evalSummaries.Count)
  }

  $collectedSummary = Join-Path $outDir 'collected\route_b_results_summary.json'
  if (Test-Path -Path $collectedSummary -PathType Leaf) {
    $lastCompleted = 'collected'
    $lastOutputTimes += (Get-Item $collectedSummary).LastWriteTime
  }
  $benchmarkSummary = Join-Path $outDir 'multisource_tracklet_policy_benchmark_summary.json'
  if (Test-Path -Path $benchmarkSummary -PathType Leaf) {
    $lastCompleted = 'benchmark_summary'
    $lastOutputTimes += (Get-Item $benchmarkSummary).LastWriteTime
  }
}

if ($preflightPath -and (Test-Path -Path $preflightPath -PathType Leaf)) {
  try {
    $preflightJson = Get-Content -Path $preflightPath -Raw | ConvertFrom-Json
    Write-Host ('PREFLIGHT={0}' -f $preflightPath)
    Write-Host ('PREFLIGHT_VALID={0}' -f $preflightJson.valid)
    Write-Host ('PREFLIGHT_TRAIN_ACTION_CHUNKS={0}' -f $preflightJson.train_action_chunk_samples)
    Write-Host ('PREFLIGHT_EVAL_ACTION_CHUNKS={0}' -f $preflightJson.eval_action_chunk_samples)
  } catch {
    Write-Host ('PREFLIGHT={0}' -f $preflightPath)
    Write-Host ('PREFLIGHT_PARSE_ERROR={0}' -f $_.Exception.Message)
  }
}

if ($stdoutPath -and (Test-Path -Path $stdoutPath -PathType Leaf)) {
  $lastOutputTimes += (Get-Item $stdoutPath).LastWriteTime
}
if ($stderrPath -and (Test-Path -Path $stderrPath -PathType Leaf)) {
  $lastOutputTimes += (Get-Item $stderrPath).LastWriteTime
}

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

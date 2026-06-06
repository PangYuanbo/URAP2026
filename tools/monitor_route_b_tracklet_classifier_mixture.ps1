param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_tracklet_classifier_mixture'),
  [string]$RunId = 'route_b_tracklet_classifier_mixture',
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
if ($process -and $process.CommandLine -like '*run-tracklet-classifier-mixture-benchmark*') {
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
$evalDatasetsRaw = Get-MetaValue 'eval_datasets'
$evalCsvsRaw = Get-MetaValue 'eval_csvs'

$evalNames = @()
if ($evalDatasetsRaw) {
  $evalNames = @($evalDatasetsRaw -split ';' | Where-Object { $_ })
}
if ($evalNames.Count -eq 0 -and $evalCsvsRaw) {
  $evalNames = @($evalCsvsRaw -split ';' | Where-Object { $_ } | ForEach-Object { [IO.Path]::GetFileNameWithoutExtension($_) })
}

$totalUnits = 1 + [Math]::Max(1, $evalNames.Count)
$doneUnits = 0
$lastCompleted = ''
$lastOutputTimes = @()

if ($preflightPath -and (Test-Path -Path $preflightPath -PathType Leaf)) {
  $lastOutputTimes += (Get-Item $preflightPath).LastWriteTime
  try {
    $preflight = Get-Content -Path $preflightPath -Raw | ConvertFrom-Json
    Write-Host ('PREFLIGHT={0}' -f $preflightPath)
    Write-Host ('PREFLIGHT_VALID={0}' -f $preflight.valid)
    Write-Host ('PREFLIGHT_TRAIN_ROWS={0}' -f $preflight.combined.train_rows)
    Write-Host ('PREFLIGHT_TRAIN_POS_NEG={0}/{1}' -f $preflight.combined.train_positives, $preflight.combined.train_negatives)
    Write-Host ('PREFLIGHT_EVAL_ROWS={0}' -f $preflight.combined.eval_rows)
    Write-Host ('PREFLIGHT_EVAL_POS_NEG={0}/{1}' -f $preflight.combined.eval_positives, $preflight.combined.eval_negatives)
    if ($preflight.errors -and $preflight.errors.Count -gt 0) {
      Write-Host 'PREFLIGHT_ERRORS='
      $preflight.errors | Select-Object -First 20
    }
  } catch {
    Write-Host ('PREFLIGHT_PARSE_ERROR={0}' -f $_.Exception.Message)
  }
}

if ($outDir -and (Test-Path -Path $outDir -PathType Container)) {
  Write-Host ('OUTPUT_DIR={0}' -f $outDir)
  $mixedManifest = Join-Path $outDir 'train\mixed_tracklets.manifest.json'
  if (Test-Path -Path $mixedManifest -PathType Leaf) {
    $doneUnits += 1
    $lastCompleted = 'train'
    $lastOutputTimes += (Get-Item $mixedManifest).LastWriteTime
  }

  $checkpoint = Join-Path $outDir 'train\joint_tracklet_classifier.pt'
  if (Test-Path -Path $checkpoint -PathType Leaf) {
    $lastCompleted = 'checkpoint'
    $lastOutputTimes += (Get-Item $checkpoint).LastWriteTime
  }

  $evalRoot = Join-Path $outDir 'eval'
  if (Test-Path -Path $evalRoot -PathType Container) {
    $evalSummaries = Get-ChildItem -Path $evalRoot -Recurse -Filter 'tracklet_classifier_threshold_summary.json' -File -ErrorAction SilentlyContinue
    foreach ($summary in $evalSummaries) {
      $doneUnits += 1
      $lastCompleted = ('eval/{0}' -f $summary.Directory.Name)
      $lastOutputTimes += $summary.LastWriteTime
    }
    Write-Host ('EVAL_SUMMARIES={0}' -f $evalSummaries.Count)
  }

  $benchmarkSummary = Join-Path $outDir 'tracklet_classifier_mixture_benchmark_summary.json'
  if (Test-Path -Path $benchmarkSummary -PathType Leaf) {
    $lastCompleted = 'benchmark_summary'
    $lastOutputTimes += (Get-Item $benchmarkSummary).LastWriteTime
    try {
      $summaryJson = Get-Content -Path $benchmarkSummary -Raw | ConvertFrom-Json
      Write-Host ('BENCHMARK_SUMMARY={0}' -f $benchmarkSummary)
      if ($summaryJson.best_by_dataset) {
        Write-Host 'BEST_BY_DATASET='
        $summaryJson.best_by_dataset | ConvertTo-Json -Depth 8
      }
    } catch {
      Write-Host ('BENCHMARK_SUMMARY_PARSE_ERROR={0}' -f $_.Exception.Message)
    }
  }
  $baselineReport = Join-Path $outDir 'baseline_report\route_b_tracklet_classifier_baseline_report.md'
  if (Test-Path -Path $baselineReport -PathType Leaf) {
    Write-Host ('BASELINE_REPORT={0}' -f $baselineReport)
    $lastOutputTimes += (Get-Item $baselineReport).LastWriteTime
  }
  $baselineComparison = Join-Path $outDir 'baseline_report\comparison\route_b_baseline_comparison_summary.json'
  if (Test-Path -Path $baselineComparison -PathType Leaf) {
    try {
      $comparisonJson = Get-Content -Path $baselineComparison -Raw | ConvertFrom-Json
      Write-Host ('BASELINE_COMPARISON={0}' -f $baselineComparison)
      Write-Host ('BASELINE_ROUTE_B_WINS={0}/{1}' -f $comparisonJson.route_b_wins, $comparisonJson.num_comparisons)
    } catch {
      Write-Host ('BASELINE_COMPARISON_PARSE_ERROR={0}' -f $_.Exception.Message)
    }
    $lastOutputTimes += (Get-Item $baselineComparison).LastWriteTime
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

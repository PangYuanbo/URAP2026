param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_tracklet_classifier_frame_benchmark'),
  [string]$RunId = 'route_b_tracklet_classifier_frame_benchmark',
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
$meta | Select-Object -First 180

$pidValue = $null
if (Test-Path -Path $pidFile -PathType Leaf) {
  $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$process = $null
if ($pidValue -match '^\d+$') {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
}

Write-Host ''
if ($process -and $process.CommandLine -like '*run-tracklet-classifier-frame-benchmark*') {
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
$datasetNamesRaw = Get-MetaValue 'dataset_names'
$runRootsRaw = Get-MetaValue 'run_roots'

if ($preflightPath) {
  Write-Host ''
  if (Test-Path -Path $preflightPath -PathType Leaf) {
    try {
      $preflightJson = Get-Content -Path $preflightPath -Raw | ConvertFrom-Json
      Write-Host ('PREFLIGHT={0}' -f $preflightPath)
      Write-Host ('PREFLIGHT_VALID={0}' -f $preflightJson.valid)
      Write-Host ('PREFLIGHT_PAIRS={0}' -f $preflightJson.combined.pairs)
      Write-Host ('PREFLIGHT_PREDICTION_ROWS={0}' -f $preflightJson.combined.prediction_rows)
      Write-Host ('PREFLIGHT_GT_BOXES={0}' -f $preflightJson.combined.gt_boxes)
      if ($preflightJson.errors -and $preflightJson.errors.Count -gt 0) {
        Write-Host 'PREFLIGHT_ERRORS='
        $preflightJson.errors | Select-Object -First 20
      }
      if ($preflightJson.warnings -and $preflightJson.warnings.Count -gt 0) {
        Write-Host 'PREFLIGHT_WARNINGS='
        $preflightJson.warnings | Select-Object -First 20
      }
    } catch {
      Write-Host ('PREFLIGHT_PARSE_ERROR={0}' -f $_.Exception.Message)
    }
  } else {
    Write-Host ('PREFLIGHT_MISSING={0}' -f $preflightPath)
  }
}

$datasetNames = @()
if ($datasetNamesRaw) {
  $datasetNames = @($datasetNamesRaw -split ';' | Where-Object { $_ })
}
if ($datasetNames.Count -eq 0 -and $runRootsRaw) {
  $datasetNames = @($runRootsRaw -split ';' | Where-Object { $_ } | ForEach-Object { [IO.Path]::GetFileName($_) })
}

$totalUnits = [Math]::Max(1, $datasetNames.Count)
$doneUnits = 0
$lastCompleted = ''
$lastOutputTimes = @()

if ($outDir -and (Test-Path -Path $outDir -PathType Container)) {
  Write-Host ('OUTPUT_DIR={0}' -f $outDir)
  $resolvedOutDir = (Resolve-Path $outDir).Path
  $datasetSummaries = Get-ChildItem -Path $outDir -Recurse -Filter 'tracklet_classifier_frame_benchmark_summary.json' -File -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Directory.FullName -ne $resolvedOutDir -and
      $_.FullName -notlike "*$([IO.Path]::DirectorySeparatorChar)threshold_*"
    }
  foreach ($summary in $datasetSummaries) {
    $doneUnits += 1
    $lastCompleted = ('dataset/{0}' -f $summary.Directory.Name)
    $lastOutputTimes += $summary.LastWriteTime
  }
  Write-Host ('DATASET_SUMMARIES={0}' -f $datasetSummaries.Count)

  $benchmarkSummary = Join-Path $outDir 'tracklet_classifier_frame_benchmark_summary.json'
  if (Test-Path -Path $benchmarkSummary -PathType Leaf) {
    $lastCompleted = 'benchmark_summary'
    $lastOutputTimes += (Get-Item $benchmarkSummary).LastWriteTime
    try {
      $summaryJson = Get-Content -Path $benchmarkSummary -Raw | ConvertFrom-Json
      Write-Host ('BENCHMARK_SUMMARY={0}' -f $benchmarkSummary)
      foreach ($dataset in $summaryJson.datasets) {
        $best = $dataset.best
        $metrics = $best.filtered_metrics
        Write-Host ('BEST dataset={0} threshold={1} f1={2} precision={3} recall={4} tp={5} fp={6} fn={7}' -f $dataset.dataset, $best.threshold, $metrics.f1, $metrics.precision, $metrics.recall, $metrics.tp, $metrics.fp, $metrics.fn)
      }
    } catch {
      Write-Host ('BENCHMARK_SUMMARY_PARSE_ERROR={0}' -f $_.Exception.Message)
    }
  }

  $baselineReport = Join-Path $outDir 'baseline_report\route_b_tracklet_classifier_frame_baseline_report.md'
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

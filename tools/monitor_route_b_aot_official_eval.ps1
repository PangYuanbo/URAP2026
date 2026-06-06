param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_official\aot_official_eval_runner'),
  [string]$RunId = 'route_b_aot_official_eval',
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
if ($process -and $process.CommandLine -like '*evaluate_aot.py*') {
  Write-Host ('RUNNING=true PID={0}' -f $pidValue)
  $pidStartText = 'unknown'
  try {
    if ($process.CreationDate -is [datetime]) {
      $pidStartText = $process.CreationDate.ToString('yyyy-MM-dd HH:mm:ss')
    } else {
      $pidStartText = ([Management.ManagementDateTimeConverter]::ToDateTime($process.CreationDate)).ToString('yyyy-MM-dd HH:mm:ss')
    }
  } catch {
    $pidStartText = [string]$process.CreationDate
  }
  Write-Host ('PID_START={0}' -f $pidStartText)
  Write-Host ('PROCESS_COMMAND={0}' -f $process.CommandLine)
} else {
  Write-Host ('NOT RUNNING PID={0}' -f $pidValue)
}

$stdoutPath = Get-MetaValue 'stdout'
$stderrPath = Get-MetaValue 'stderr'
$evaluationFolder = Get-MetaValue 'evaluation_folder'
$resultsFolder = Get-MetaValue 'results_folder'
$predictionParts = Get-MetaValue 'prediction_parts'
$preflightPath = Get-MetaValue 'preflight'

$doneUnits = 0
$totalUnits = 1
$lastCompleted = ''
$lastOutputTimes = @()

if ($preflightPath) {
  Write-Host ''
  if (Test-Path -Path $preflightPath -PathType Leaf) {
    $lastOutputTimes += (Get-Item $preflightPath).LastWriteTime
    try {
      $preflightJson = Get-Content -Path $preflightPath -Raw | ConvertFrom-Json
      Write-Host ('PREFLIGHT={0}' -f $preflightPath)
      Write-Host ('PREFLIGHT_VALID={0}' -f $preflightJson.valid)
      Write-Host ('PREFLIGHT_PARTS={0}' -f $preflightJson.combined.parts)
      Write-Host ('PREFLIGHT_RECORDS_CHECKED={0}' -f $preflightJson.combined.records_checked)
      Write-Host ('PREFLIGHT_DETECTIONS_CHECKED={0}' -f $preflightJson.combined.detections_checked)
      Write-Host ('PREFLIGHT_PATTERN_ERRORS={0}' -f $preflightJson.combined.pattern_errors)
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

if ($resultsFolder -and (Test-Path -Path $resultsFolder -PathType Container)) {
  $parts = @(Get-ChildItem -Path $resultsFolder -Filter '*.pkl' -File -ErrorAction SilentlyContinue)
  Write-Host ('AOT_PREDICTION_PARTS={0}' -f $parts.Count)
  foreach ($part in $parts) { $lastOutputTimes += $part.LastWriteTime }
}
if ($predictionParts) {
  Write-Host ('AOT_PREDICTION_PARTS_AT_START={0}' -f $predictionParts)
}

if ($evaluationFolder -and (Test-Path -Path $evaluationFolder -PathType Container)) {
  Write-Host ('EVALUATION_FOLDER={0}' -f $evaluationFolder)
  $resultJson = Join-Path $evaluationFolder 'result\result.json'
  if (Test-Path -Path $resultJson -PathType Leaf) {
    Write-Host ('AOT_RESULT_JSON={0}' -f $resultJson)
    $lastCompleted = 'result_json'
    $lastOutputTimes += (Get-Item $resultJson).LastWriteTime
  }
  $summaryFiles = @(Get-ChildItem -Path (Join-Path $evaluationFolder 'summaries') -Filter 'result_metrics*_summary*.json' -File -ErrorAction SilentlyContinue)
  if ($summaryFiles.Count -gt 0) {
    $latestSummary = $summaryFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    Write-Host ('AOT_SUMMARY={0}' -f $latestSummary.FullName)
    $lastCompleted = 'official_metrics_summary'
    $doneUnits = 1
    $lastOutputTimes += $latestSummary.LastWriteTime
    try {
      $summaryJson = Get-Content -Path $latestSummary.FullName -Raw | ConvertFrom-Json
      $flat = $summaryJson | ConvertTo-Json -Compress
      Write-Host ('AOT_SUMMARY_JSON_COMPACT={0}' -f $flat)
    } catch {
      Write-Host ('AOT_SUMMARY_PARSE_ERROR={0}' -f $_.Exception.Message)
    }
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
if ($stderrPath -and (Test-Path $stderrPath -PathType Leaf)) {
  $stderrTail = Get-Content -Path $stderrPath -Tail $TailLines -ErrorAction SilentlyContinue
  if ($stderrTail) {
    Write-Host ''
    Write-Host '== stderr tail =='
    $stderrTail
  }
}

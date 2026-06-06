param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string[]]$RunRoots = @(),
  [string[]]$GtCsvs = @(),
  [string]$Weights = '',
  [string]$OutDir = (Join-Path $RepoRoot 'artifacts\route_b_tracklet_classifier_frame_benchmark\run'),
  [string[]]$DatasetNames = @(),
  [string]$PredictionName = 'predictions.jsonl',
  [string]$DiagnosticsName = 'diagnostics.jsonl',
  [double]$Threshold = 0.5,
  [double[]]$Thresholds = @(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
  [string]$UntrackedPolicy = 'keep',
  [switch]$DisableTrackletPromotion,
  [double]$PromotionScoreFloor = 0.22,
  [double]$PromotionMinBranchDrone = 0.40,
  [double]$PromotionMaxBackground = 0.68,
  [switch]$SelectivePromotion,
  [double]$SelectiveMinTemporalCropDelta = 0.05,
  [double]$SelectiveMinTemporalBackgroundMargin = -0.05,
  [double]$SelectiveMaxTrackletBackground = 0.60,
  [double]$SelectiveMaxTrackletObjectness = 0.50,
  [int]$SelectiveMinTrackletRows = 2,
  [double]$SelectiveMinTemporalGainRate = 0.40,
  [double]$SelectiveMinWeakDetectorTemporalSignal = 0.05,
  [switch]$SelectiveAllowNonRecoverySource,
  [int]$SelectiveMaxPromotedTrackletsPerSequence = 2,
  [double]$IouThreshold = 0.3,
  [double]$ScoreThreshold = 0.0,
  [Nullable[int]]$MaxFrames = $null,
  [string]$BaselineCsv = '',
  [string]$BaselineMetric = 'frame_best_f1',
  [switch]$BaselineLowerIsBetter,
  [int]$BaselineDigits = 3,
  [switch]$AllowInvalidBaselines,
  [switch]$SkipPreflight,
  [switch]$AllowInvalidPreflight,
  [string]$PreflightOut = '',
  [string]$RunId = 'route_b_tracklet_classifier_frame_benchmark',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_tracklet_classifier_frame_benchmark')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not $Weights) { throw 'Weights must be provided' }
if (-not (Test-Path -Path $Weights -PathType Leaf)) { throw "Weights not found: $Weights" }
if ($RunRoots.Count -lt 1) { throw 'RunRoots must contain at least one inference output root' }
if ($GtCsvs.Count -ne $RunRoots.Count) { throw 'GtCsvs must have the same length as RunRoots' }
if ($DatasetNames.Count -gt 0 -and $DatasetNames.Count -ne $RunRoots.Count) {
  throw 'DatasetNames must be empty or have the same length as RunRoots'
}
if ($UntrackedPolicy -notin @('keep', 'suppress')) { throw 'UntrackedPolicy must be keep or suppress' }
foreach ($path in $RunRoots) {
  if (-not (Test-Path -Path $path -PathType Container)) { throw "Run root not found: $path" }
}
foreach ($path in $GtCsvs) {
  if (-not (Test-Path -Path $path -PathType Leaf)) { throw "GT CSV not found: $path" }
}
if ($BaselineCsv -and -not (Test-Path -Path $BaselineCsv -PathType Leaf)) {
  throw "BaselineCsv not found: $BaselineCsv"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)
if (-not $PreflightOut) {
  $PreflightOut = Join-Path $OutputRoot ("runner_{0}_preflight.json" -f $RunId)
}

function ConvertTo-WindowsArgumentString([string[]]$Values) {
  $quoted = foreach ($arg in $Values) {
    if ($arg -match '[\s"]') {
      $escaped = $arg -replace '"', '\"'
      '"' + $escaped + '"'
    } else {
      $arg
    }
  }
  return ($quoted -join ' ')
}

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*run-tracklet-classifier-frame-benchmark*') {
      Write-Host "Route B tracklet classifier frame benchmark already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 180 }
      exit 0
    }
  }
}

if (-not $SkipPreflight) {
  $preflightArgs = @(
    '-m', 'qstr_dronedet.cli', 'validate-tracklet-classifier-frame-benchmark-inputs',
    '--run-roots'
  )
  $preflightArgs += $RunRoots
  $preflightArgs += '--gt-csvs'
  $preflightArgs += $GtCsvs
  $preflightArgs += @(
    '--weights', $Weights,
    '--out', $PreflightOut,
    '--prediction-name', $PredictionName,
    '--diagnostics-name', $DiagnosticsName
  )
  if ($DatasetNames.Count -gt 0) {
    $preflightArgs += '--dataset-names'
    $preflightArgs += $DatasetNames
  }
  if ($Thresholds.Count -gt 0) {
    $preflightArgs += '--thresholds'
    $preflightArgs += ($Thresholds | ForEach-Object { [string]$_ })
  }
  if ($null -ne $MaxFrames) {
    $preflightArgs += @('--max-frames', [string]$MaxFrames)
  }
  if ($BaselineCsv) {
    $preflightArgs += @(
      '--baseline-csv', $BaselineCsv,
      '--baseline-metric', $BaselineMetric
    )
  }
  if ($AllowInvalidPreflight) { $preflightArgs += '--allow-invalid' }

  $preflightArgumentString = ConvertTo-WindowsArgumentString -Values $preflightArgs
  Write-Host "Running Route B frame benchmark preflight: $PreflightOut"
  $preflightProcess = Start-Process `
    -FilePath $PythonExe `
    -ArgumentList $preflightArgumentString `
    -WorkingDirectory $RepoRoot `
    -NoNewWindow `
    -Wait `
    -PassThru
  if ($preflightProcess.ExitCode -ne 0) {
    throw "Route B frame benchmark preflight failed with exit code $($preflightProcess.ExitCode): $PreflightOut"
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

$argList = @(
  '-m', 'qstr_dronedet.cli', 'run-tracklet-classifier-frame-benchmark',
  '--run-roots'
)
$argList += $RunRoots
$argList += '--gt-csvs'
$argList += $GtCsvs
$argList += @(
  '--weights', $Weights,
  '--out-dir', $OutDir,
  '--prediction-name', $PredictionName,
  '--diagnostics-name', $DiagnosticsName,
  '--threshold', [string]$Threshold,
  '--untracked-policy', $UntrackedPolicy,
  '--promotion-score-floor', [string]$PromotionScoreFloor,
  '--promotion-min-branch-drone', [string]$PromotionMinBranchDrone,
  '--promotion-max-background', [string]$PromotionMaxBackground,
  '--selective-min-temporal-crop-delta', [string]$SelectiveMinTemporalCropDelta,
  '--selective-min-temporal-background-margin', [string]$SelectiveMinTemporalBackgroundMargin,
  '--selective-max-tracklet-background', [string]$SelectiveMaxTrackletBackground,
  '--selective-max-tracklet-objectness', [string]$SelectiveMaxTrackletObjectness,
  '--selective-min-tracklet-rows', [string]$SelectiveMinTrackletRows,
  '--selective-min-temporal-gain-rate', [string]$SelectiveMinTemporalGainRate,
  '--selective-min-weak-detector-temporal-signal', [string]$SelectiveMinWeakDetectorTemporalSignal,
  '--selective-max-promoted-tracklets-per-sequence', [string]$SelectiveMaxPromotedTrackletsPerSequence,
  '--iou-threshold', [string]$IouThreshold,
  '--score-threshold', [string]$ScoreThreshold
)
if ($DatasetNames.Count -gt 0) {
  $argList += '--dataset-names'
  $argList += $DatasetNames
}
if ($Thresholds.Count -gt 0) {
  $argList += '--thresholds'
  $argList += ($Thresholds | ForEach-Object { [string]$_ })
}
if ($DisableTrackletPromotion) { $argList += '--disable-tracklet-promotion' }
if ($SelectivePromotion) { $argList += '--selective-promotion' }
if ($SelectiveAllowNonRecoverySource) { $argList += '--selective-allow-non-recovery-source' }
if ($null -ne $MaxFrames) {
  $argList += @('--max-frames', [string]$MaxFrames)
}
if ($BaselineCsv) {
  $argList += @(
    '--baseline-csv', $BaselineCsv,
    '--baseline-metric', $BaselineMetric,
    '--baseline-digits', [string]$BaselineDigits
  )
  if ($BaselineLowerIsBetter) { $argList += '--baseline-lower-is-better' }
  if ($AllowInvalidBaselines) { $argList += '--allow-invalid-baselines' }
}

$argumentString = ConvertTo-WindowsArgumentString -Values $argList

$process = Start-Process `
  -FilePath $PythonExe `
  -ArgumentList $argumentString `
  -WorkingDirectory $RepoRoot `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

$process.Id | Set-Content -Encoding ascii -Path $pidFile

@(
  ('started={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
  ('pid={0}' -f $process.Id),
  ('python={0}' -f $PythonExe),
  ('run_id={0}' -f $RunId),
  ('repo_root={0}' -f $RepoRoot),
  ('out_dir={0}' -f $OutDir),
  ('output_root={0}' -f $OutputRoot),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('preflight={0}' -f $PreflightOut),
  ('skip_preflight={0}' -f [bool]$SkipPreflight),
  ('run_roots={0}' -f ($RunRoots -join ';')),
  ('gt_csvs={0}' -f ($GtCsvs -join ';')),
  ('dataset_names={0}' -f ($DatasetNames -join ';')),
  ('weights={0}' -f $Weights),
  ('thresholds={0}' -f ($Thresholds -join ';')),
  ('baseline_csv={0}' -f $BaselineCsv),
  ('baseline_metric={0}' -f $BaselineMetric),
  ('cmd_args={0}' -f $argumentString)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached Route B tracklet classifier frame benchmark.'
Get-Content $metaFile

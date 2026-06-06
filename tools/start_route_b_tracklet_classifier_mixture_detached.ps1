param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string[]]$TrainCsvs = @(),
  [string[]]$EvalCsvs = @(),
  [string]$OutDir = (Join-Path $RepoRoot 'artifacts\route_b_tracklet_classifier_mixture\run'),
  [string[]]$TrainSourceNames = @(),
  [string[]]$EvalDatasetNames = @(),
  [int]$Epochs = 50,
  [double]$Lr = 0.001,
  [int]$Hidden = 64,
  [int]$HardTinyPositiveAugments = 0,
  [string[]]$BalanceBy = @('dataset_source', 'bucket', 'label'),
  [double[]]$Thresholds = @(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
  [string]$BaselineCsv = '',
  [string]$BaselineMetric = 'tracklet_best_f1',
  [switch]$BaselineLowerIsBetter,
  [int]$BaselineDigits = 3,
  [switch]$AllowInvalidBaselines,
  [switch]$SkipPreflight,
  [switch]$AllowInvalidPreflight,
  [switch]$AllowTrainEvalSequenceOverlap,
  [string]$RunId = 'route_b_tracklet_classifier_mixture',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_tracklet_classifier_mixture')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if ($TrainCsvs.Count -lt 1) { throw 'TrainCsvs must contain at least one classifier CSV' }
if ($EvalCsvs.Count -lt 1) { throw 'EvalCsvs must contain at least one held-out classifier CSV' }
if ($TrainSourceNames.Count -gt 0 -and $TrainSourceNames.Count -ne $TrainCsvs.Count) {
  throw 'TrainSourceNames must be empty or have the same length as TrainCsvs'
}
if ($EvalDatasetNames.Count -gt 0 -and $EvalDatasetNames.Count -ne $EvalCsvs.Count) {
  throw 'EvalDatasetNames must be empty or have the same length as EvalCsvs'
}
foreach ($path in $TrainCsvs) {
  if (-not (Test-Path -Path $path -PathType Leaf)) { throw "Train CSV not found: $path" }
}
foreach ($path in $EvalCsvs) {
  if (-not (Test-Path -Path $path -PathType Leaf)) { throw "Eval CSV not found: $path" }
}
if ($BaselineCsv -and -not (Test-Path -Path $BaselineCsv -PathType Leaf)) {
  throw "BaselineCsv not found: $BaselineCsv"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)
$preflightOut = Join-Path $OutDir 'tracklet_classifier_mixture_preflight.json'

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*run-tracklet-classifier-mixture-benchmark*') {
      Write-Host "Route B tracklet classifier mixture already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 160 }
      exit 0
    }
  }
}

if (-not $SkipPreflight) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $preflightOut) | Out-Null
  $preflightArgs = @(
    '-m', 'qstr_dronedet.cli', 'validate-tracklet-classifier-mixture-inputs',
    '--train-csvs'
  )
  $preflightArgs += $TrainCsvs
  $preflightArgs += '--eval-csvs'
  $preflightArgs += $EvalCsvs
  $preflightArgs += @('--out', $preflightOut)
  if ($TrainSourceNames.Count -gt 0) {
    $preflightArgs += '--train-source-names'
    $preflightArgs += $TrainSourceNames
  }
  if ($EvalDatasetNames.Count -gt 0) {
    $preflightArgs += '--eval-dataset-names'
    $preflightArgs += $EvalDatasetNames
  }
  if ($AllowTrainEvalSequenceOverlap) {
    $preflightArgs += '--allow-train-eval-sequence-overlap'
  }
  if ($AllowInvalidPreflight) {
    $preflightArgs += '--allow-invalid'
  }
  & $PythonExe $preflightArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Tracklet classifier mixture preflight failed. Report: $preflightOut"
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

$argList = @(
  '-m', 'qstr_dronedet.cli', 'run-tracklet-classifier-mixture-benchmark',
  '--train-csvs'
)
$argList += $TrainCsvs
$argList += '--eval-csvs'
$argList += $EvalCsvs
$argList += @(
  '--out-dir', $OutDir,
  '--epochs', [string]$Epochs,
  '--lr', [string]$Lr,
  '--hidden', [string]$Hidden,
  '--hard-tiny-positive-augments', [string]$HardTinyPositiveAugments
)
if ($TrainSourceNames.Count -gt 0) {
  $argList += '--train-source-names'
  $argList += $TrainSourceNames
}
if ($EvalDatasetNames.Count -gt 0) {
  $argList += '--eval-dataset-names'
  $argList += $EvalDatasetNames
}
if ($BalanceBy.Count -gt 0) {
  $argList += '--balance-by'
  $argList += $BalanceBy
}
if ($Thresholds.Count -gt 0) {
  $argList += '--thresholds'
  $argList += ($Thresholds | ForEach-Object { [string]$_ })
}
if ($BaselineCsv) {
  $argList += @(
    '--baseline-csv', $BaselineCsv,
    '--baseline-metric', $BaselineMetric,
    '--baseline-digits', [string]$BaselineDigits
  )
  if ($BaselineLowerIsBetter) {
    $argList += '--baseline-lower-is-better'
  }
  if ($AllowInvalidBaselines) {
    $argList += '--allow-invalid-baselines'
  }
}
if ($SkipPreflight) {
  $argList += '--skip-preflight'
}
if ($AllowInvalidPreflight) {
  $argList += '--allow-invalid-preflight'
}
if ($AllowTrainEvalSequenceOverlap) {
  $argList += '--allow-train-eval-sequence-overlap'
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
  ('preflight={0}' -f $preflightOut),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('train_csvs={0}' -f ($TrainCsvs -join ';')),
  ('eval_csvs={0}' -f ($EvalCsvs -join ';')),
  ('train_sources={0}' -f ($TrainSourceNames -join ';')),
  ('eval_datasets={0}' -f ($EvalDatasetNames -join ';')),
  ('epochs={0}' -f $Epochs),
  ('hidden={0}' -f $Hidden),
  ('balance_by={0}' -f ($BalanceBy -join ';')),
  ('baseline_csv={0}' -f $BaselineCsv),
  ('baseline_metric={0}' -f $BaselineMetric),
  ('cmd_args={0}' -f $argumentString)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached Route B tracklet classifier mixture benchmark.'
Get-Content $metaFile

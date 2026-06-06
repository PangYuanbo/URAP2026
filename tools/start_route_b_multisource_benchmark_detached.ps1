param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string[]]$TrainTrackletInputs = @(),
  [string[]]$EvalTrackletInputs = @(),
  [string]$OutDir = (Join-Path $RepoRoot 'artifacts\route_b_multisource_benchmark\run'),
  [string[]]$TrainSourceNames = @(),
  [string[]]$EvalDatasetNames = @(),
  [int]$PastLen = 8,
  [int]$FutureLen = 8,
  [Nullable[int]]$ImageWidth = $null,
  [Nullable[int]]$ImageHeight = $null,
  [switch]$PositivesOnly,
  [int]$MinTrackletRows = 0,
  [double]$CalibFraction = 0.2,
  [double]$TestFraction = 0.0,
  [int]$Seed = 59,
  [string]$GroupField = 'seq',
  [string]$SourceField = 'dataset_source',
  [string[]]$ModelTypes = @('mlp', 'diffusion'),
  [int]$Epochs = 50,
  [double]$Lr = 0.001,
  [int]$Hidden = 128,
  [int]$BatchSize = 64,
  [int]$DiffusionSteps = 16,
  [double]$ErrorScale = 8.0,
  [double[]]$Thresholds = @(),
  [string[]]$BalanceBy = @('dataset_source'),
  [string]$BaselineCsv = '',
  [string]$BaselineMetric = 'best_f1',
  [switch]$BaselineLowerIsBetter,
  [int]$BaselineDigits = 3,
  [switch]$AllowInvalidBaselines,
  [switch]$SkipPreflight,
  [string]$PreflightOut = '',
  [string]$RunId = 'route_b_multisource_benchmark',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_multisource_benchmark')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if ($TrainTrackletInputs.Count -lt 1) { throw 'TrainTrackletInputs must contain at least one tracklet JSONL' }
if ($EvalTrackletInputs.Count -lt 1) { throw 'EvalTrackletInputs must contain at least one held-out tracklet JSONL' }
if ($TrainSourceNames.Count -gt 0 -and $TrainSourceNames.Count -ne $TrainTrackletInputs.Count) {
  throw 'TrainSourceNames must be empty or have the same length as TrainTrackletInputs'
}
if ($EvalDatasetNames.Count -gt 0 -and $EvalDatasetNames.Count -ne $EvalTrackletInputs.Count) {
  throw 'EvalDatasetNames must be empty or have the same length as EvalTrackletInputs'
}
if (($null -eq $ImageWidth) -xor ($null -eq $ImageHeight)) {
  throw 'ImageWidth and ImageHeight must be provided together'
}
foreach ($path in $TrainTrackletInputs) {
  if (-not (Test-Path -Path $path -PathType Leaf)) { throw "Train tracklet input not found: $path" }
}
foreach ($path in $EvalTrackletInputs) {
  if (-not (Test-Path -Path $path -PathType Leaf)) { throw "Eval tracklet input not found: $path" }
}
if ($BaselineCsv -and -not (Test-Path -Path $BaselineCsv -PathType Leaf)) {
  throw "BaselineCsv not found: $BaselineCsv"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

if (-not $PreflightOut) {
  $PreflightOut = Join-Path $OutputRoot ("runner_{0}_preflight.json" -f $RunId)
}

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*run-multisource-tracklet-policy-benchmark*') {
      Write-Host "Route B multisource benchmark already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 140 }
      exit 0
    }
  }
}

if (-not $SkipPreflight) {
  $preflightArgs = @(
    '-m', 'qstr_dronedet.cli', 'validate-route-b-tracklet-inputs',
    '--train-tracklet-inputs'
  )
  $preflightArgs += $TrainTrackletInputs
  $preflightArgs += '--eval-tracklet-inputs'
  $preflightArgs += $EvalTrackletInputs
  $preflightArgs += @(
    '--out', $PreflightOut,
    '--past-len', [string]$PastLen,
    '--future-len', [string]$FutureLen,
    '--min-tracklet-rows', [string]$MinTrackletRows,
    '--strict'
  )
  if ($TrainSourceNames.Count -gt 0) {
    $preflightArgs += '--train-source-names'
    $preflightArgs += $TrainSourceNames
  }
  if ($EvalDatasetNames.Count -gt 0) {
    $preflightArgs += '--eval-dataset-names'
    $preflightArgs += $EvalDatasetNames
  }
  & $PythonExe $preflightArgs
  if ($LASTEXITCODE -ne 0) {
    throw "Route B preflight failed. Report: $PreflightOut"
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

$argList = @(
  '-m', 'qstr_dronedet.cli', 'run-multisource-tracklet-policy-benchmark',
  '--train-tracklet-inputs'
)
$argList += $TrainTrackletInputs
$argList += '--eval-tracklet-inputs'
$argList += $EvalTrackletInputs
$argList += @(
  '--out-dir', $OutDir,
  '--past-len', [string]$PastLen,
  '--future-len', [string]$FutureLen,
  '--min-tracklet-rows', [string]$MinTrackletRows,
  '--calib-fraction', [string]$CalibFraction,
  '--test-fraction', [string]$TestFraction,
  '--seed', [string]$Seed,
  '--group-field', $GroupField,
  '--source-field', $SourceField,
  '--model-types'
)
$argList += $ModelTypes
$argList += @(
  '--epochs', [string]$Epochs,
  '--lr', [string]$Lr,
  '--hidden', [string]$Hidden,
  '--batch-size', [string]$BatchSize,
  '--diffusion-steps', [string]$DiffusionSteps,
  '--error-scale', [string]$ErrorScale
)
if ($TrainSourceNames.Count -gt 0) {
  $argList += '--train-source-names'
  $argList += $TrainSourceNames
}
if ($EvalDatasetNames.Count -gt 0) {
  $argList += '--eval-dataset-names'
  $argList += $EvalDatasetNames
}
if ($null -ne $ImageWidth) {
  $argList += @('--image-width', [string]$ImageWidth, '--image-height', [string]$ImageHeight)
}
if ($PositivesOnly) {
  $argList += '--positives-only'
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
if ($BalanceBy.Count -gt 0) {
  $argList += '--balance-by'
  $argList += $BalanceBy
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
  ('preflight={0}' -f $PreflightOut),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('train_inputs={0}' -f ($TrainTrackletInputs -join ';')),
  ('eval_inputs={0}' -f ($EvalTrackletInputs -join ';')),
  ('eval_datasets={0}' -f ($EvalDatasetNames -join ';')),
  ('baseline_csv={0}' -f $BaselineCsv),
  ('baseline_metric={0}' -f $BaselineMetric),
  ('cmd_args={0}' -f $argumentString)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached Route B multisource tracklet policy benchmark.'
Get-Content $metaFile

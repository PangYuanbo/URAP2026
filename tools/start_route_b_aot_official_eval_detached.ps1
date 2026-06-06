param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$TransVisDroneRepo = (Join-Path $RepoRoot 'papers\TransVisDrone'),
  [string]$PythonExe = (Join-Path $RepoRoot 'papers\TransVisDrone\.venv\Scripts\python.exe'),
  [string]$ResultsFolder = '',
  [string]$EvaluationFolder = (Join-Path $RepoRoot 'artifacts\route_b_official\aot_official_eval'),
  [string]$DatasetPath = 'D:\URAP_datasets\AOT\part1',
  [double]$DetectionThreshold = 0.2,
  [string]$ClipIdToFlightIdPath = '',
  [switch]$SkipPreflight,
  [switch]$AllowInvalidPreflight,
  [string]$PreflightOut = '',
  [string]$RunId = 'route_b_aot_official_eval',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\route_b_official\aot_official_eval_runner')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $TransVisDroneRepo -PathType Container)) { throw "TransVisDroneRepo not found: $TransVisDroneRepo" }
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not $ResultsFolder) { throw 'ResultsFolder must be provided and point to aotpredictions' }
if (-not (Test-Path -Path $ResultsFolder -PathType Container)) { throw "ResultsFolder not found: $ResultsFolder" }
if (-not (Test-Path -Path $DatasetPath -PathType Container)) { throw "DatasetPath not found: $DatasetPath" }

$ResultsFolder = (Resolve-Path $ResultsFolder).Path
$DatasetPath = (Resolve-Path $DatasetPath).Path
if (-not [System.IO.Path]::IsPathRooted($EvaluationFolder)) {
  $EvaluationFolder = Join-Path (Get-Location) $EvaluationFolder
}
$EvaluationFolder = [System.IO.Path]::GetFullPath($EvaluationFolder)

$evalScript = Join-Path $TransVisDroneRepo 'evaluate_aot.py'
if (-not (Test-Path -Path $evalScript -PathType Leaf)) { throw "evaluate_aot.py not found: $evalScript" }
if (-not $ClipIdToFlightIdPath) {
  $ClipIdToFlightIdPath = Join-Path $TransVisDroneRepo 'aot_flight_ids\aot_clip_id_to_flight_id.pkl'
}
if ($ClipIdToFlightIdPath -and -not (Test-Path -Path $ClipIdToFlightIdPath -PathType Leaf)) {
  throw "ClipIdToFlightIdPath not found: $ClipIdToFlightIdPath"
}

$predictionParts = @(Get-ChildItem -Path $ResultsFolder -Filter '*.pkl' -File -ErrorAction SilentlyContinue)
if ($predictionParts.Count -lt 1) { throw "No .pkl prediction parts found in ResultsFolder: $ResultsFolder" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $EvaluationFolder) | Out-Null
if (-not $PreflightOut) {
  $PreflightOut = Join-Path $OutputRoot ("runner_{0}_preflight.json" -f $RunId)
}

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*evaluate_aot.py*') {
      Write-Host "Route B AOT official eval already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 160 }
      exit 0
    }
  }
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

if (-not $SkipPreflight) {
  $preflightArgs = @(
    '-m', 'qstr_dronedet.cli', 'validate-tracklet-classifier-aot-eval-inputs',
    '--results-folder', $ResultsFolder,
    '--out', $PreflightOut,
    '--clip-id-to-flight-id-path', $ClipIdToFlightIdPath
  )
  if ($AllowInvalidPreflight) { $preflightArgs += '--allow-invalid' }
  $preflightArgumentString = ConvertTo-WindowsArgumentString -Values $preflightArgs
  Write-Host "Running Route B AOT eval preflight: $PreflightOut"
  $preflightProcess = Start-Process `
    -FilePath (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe') `
    -ArgumentList $preflightArgumentString `
    -WorkingDirectory $RepoRoot `
    -NoNewWindow `
    -Wait `
    -PassThru
  if ($preflightProcess.ExitCode -ne 0) {
    throw "Route B AOT eval preflight failed with exit code $($preflightProcess.ExitCode): $PreflightOut"
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

$argList = @(
  '.\evaluate_aot.py',
  '--results_folder', $ResultsFolder,
  '--evaluation_folder', $EvaluationFolder,
  '--detection_threshold', [string]$DetectionThreshold,
  '--dataset-path', $DatasetPath
)
$argumentString = ConvertTo-WindowsArgumentString -Values $argList

$process = Start-Process `
  -FilePath $PythonExe `
  -ArgumentList $argumentString `
  -WorkingDirectory $TransVisDroneRepo `
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
  ('transvisdrone_repo={0}' -f $TransVisDroneRepo),
  ('results_folder={0}' -f $ResultsFolder),
  ('prediction_parts={0}' -f $predictionParts.Count),
  ('preflight={0}' -f $PreflightOut),
  ('skip_preflight={0}' -f [bool]$SkipPreflight),
  ('clip_id_to_flight_id_path={0}' -f $ClipIdToFlightIdPath),
  ('evaluation_folder={0}' -f $EvaluationFolder),
  ('dataset_path={0}' -f $DatasetPath),
  ('detection_threshold={0}' -f $DetectionThreshold),
  ('output_root={0}' -f $OutputRoot),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('cmd_args={0}' -f $argumentString)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached Route B AOT official evaluation.'
Get-Content $metaFile

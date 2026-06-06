param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$RunRoot,
  [string]$GtCsv,
  [string]$OutDir,
  [string]$Profile = 'hard_recovery',
  [string]$DiagnosticsName = 'diagnostics_raw.jsonl',
  [int]$MaxGap = 3,
  [double]$BaseRadius = 18.0,
  [double]$RadiusPerSide = 0.75,
  [double]$MinIou = 0.05,
  [double]$MinScore = 0.0,
  [int]$MinTrackletRows = 3,
  [double]$IouThreshold = 0.5,
  [double]$CenterThreshold = 24.0,
  [string]$RunId = 'yolomg_build_proposal_tracklets',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\yolomg_build_proposal_tracklets_runner')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path -Path $RunRoot -PathType Container)) { throw "RunRoot not found: $RunRoot" }
if (-not (Test-Path -Path $GtCsv -PathType Leaf)) { throw "GtCsv not found: $GtCsv" }
if (-not $OutDir) { throw 'OutDir must be provided' }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*build-proposal-tracklet-dataset*') {
      Write-Host "Proposal-tracklet build already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)
$argList = @(
  '-m', 'qstr_dronedet.cli', 'build-proposal-tracklet-dataset',
  '--run-roots', $RunRoot,
  '--gt-csv', $GtCsv,
  '--out', $OutDir,
  '--profile', $Profile,
  '--diagnostics-name', $DiagnosticsName,
  '--max-gap', [string]$MaxGap,
  '--base-radius', [string]$BaseRadius,
  '--radius-per-side', [string]$RadiusPerSide,
  '--min-iou', [string]$MinIou,
  '--min-score', [string]$MinScore,
  '--min-tracklet-rows', [string]$MinTrackletRows,
  '--iou-threshold', [string]$IouThreshold,
  '--center-threshold', [string]$CenterThreshold
)

$env:PYTHONPATH = $RepoRoot
$proc = Start-Process -FilePath $PythonExe -ArgumentList $argList -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$proc.Id | Set-Content -Encoding ascii -Path $pidFile
@(
  ('started={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
  ('pid={0}' -f $proc.Id),
  ('python={0}' -f $PythonExe),
  ('run_id={0}' -f $RunId),
  ('run_root={0}' -f $RunRoot),
  ('gt_csv={0}' -f $GtCsv),
  ('out_dir={0}' -f $OutDir),
  ('profile={0}' -f $Profile),
  ('diagnostics_name={0}' -f $DiagnosticsName),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('cmd_args={0}' -f ($argList -join ' '))
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached YOLOMG proposal-tracklet build.'
Get-Content $metaFile

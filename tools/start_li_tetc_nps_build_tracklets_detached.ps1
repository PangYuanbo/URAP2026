param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$RunRoot = '',
  [string]$GtCsv = '',
  [string]$OutDir = '',
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
  [string]$RunId = 'li_tetc_nps_build_tracklets',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\nps_li_tetc_compare\build_tracklets_runner')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -Path $RepoRoot -PathType Container)) { throw "RepoRoot not found: $RepoRoot" }
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }

$defaultRoot = Join-Path $RepoRoot 'artifacts\nps_li_tetc_compare\li_frcnn_lowconf_proposals'
if (-not $RunRoot) { $RunRoot = Join-Path $defaultRoot 'run_root' }
if (-not $GtCsv) { $GtCsv = Join-Path $defaultRoot 'li_tetc_gt.csv' }
if (-not $OutDir) { $OutDir = Join-Path $RepoRoot 'artifacts\nps_li_tetc_compare\li_frcnn_lowconf_tracklets' }
if (-not (Test-Path -Path $RunRoot -PathType Container)) { throw "RunRoot not found: $RunRoot" }
if (-not (Test-Path -Path $GtCsv -PathType Leaf)) { throw "GtCsv not found: $GtCsv" }

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
      Write-Host "Li-TETC NPS tracklet build already running: pid=$existingPid"
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
  ('repo_root={0}' -f $RepoRoot),
  ('run_root={0}' -f $RunRoot),
  ('gt_csv={0}' -f $GtCsv),
  ('out_dir={0}' -f $OutDir),
  ('profile={0}' -f $Profile),
  ('diagnostics_name={0}' -f $DiagnosticsName),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('cmd_args={0}' -f ($argList -join ' '))
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached Li-TETC NPS proposal-tracklet build.'
Get-Content $metaFile

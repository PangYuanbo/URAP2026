param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$ImagesList,
  [string]$PredLabelDir,
  [string]$TrackletJsonl,
  [string]$OutLabelDir,
  [string]$ScoreField = 'video_action_model_fusion_score',
  [switch]$InvertScore,
  [double]$Center = 0.20,
  [double]$Beta = 0.40,
  [ValidateSet('additive', 'suppress-only', 'boost-only')]
  [string]$Mode = 'additive',
  [ValidateSet('keep', 'drop')]
  [string]$MissingScoreBehavior = 'keep',
  [int]$MinTrackletRows = 1,
  [double]$ClipMin = 0.0,
  [double]$ClipMax = 1.0,
  [string]$RunId = 'yolomg_rescore_pred_labels',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\yolomg_action\rescore_detached')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path -Path $ImagesList -PathType Leaf)) { throw "ImagesList not found: $ImagesList" }
if (-not (Test-Path -Path $PredLabelDir -PathType Container)) { throw "PredLabelDir not found: $PredLabelDir" }
if (-not (Test-Path -Path $TrackletJsonl -PathType Leaf)) { throw "TrackletJsonl not found: $TrackletJsonl" }
if (-not $OutLabelDir) { throw 'OutLabelDir must be provided' }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutLabelDir | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*yolomg_rescore_pred_labels_from_tracklets.py*') {
      Write-Host "YOLOMG pred-label rescore already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)
$script = Join-Path $RepoRoot 'tools\yolomg_rescore_pred_labels_from_tracklets.py'
$argList = @(
  $script,
  '--images-list', $ImagesList,
  '--pred-label-dir', $PredLabelDir,
  '--tracklet-jsonl', $TrackletJsonl,
  '--out-label-dir', $OutLabelDir,
  '--score-field', $ScoreField,
  '--center', [string]$Center,
  '--beta', [string]$Beta,
  '--mode', $Mode,
  '--missing-score-behavior', $MissingScoreBehavior,
  '--min-tracklet-rows', [string]$MinTrackletRows,
  '--clip-min', [string]$ClipMin,
  '--clip-max', [string]$ClipMax
)
if ($InvertScore) { $argList += '--invert-score' }

$proc = Start-Process -FilePath $PythonExe -ArgumentList $argList -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$proc.Id | Set-Content -Encoding ascii -Path $pidFile
@(
  ('started={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
  ('pid={0}' -f $proc.Id),
  ('python={0}' -f $PythonExe),
  ('run_id={0}' -f $RunId),
  ('images_list={0}' -f $ImagesList),
  ('pred_label_dir={0}' -f $PredLabelDir),
  ('tracklet_jsonl={0}' -f $TrackletJsonl),
  ('out_label_dir={0}' -f $OutLabelDir),
  ('score_field={0}' -f $ScoreField),
  ('invert_score={0}' -f $InvertScore),
  ('center={0}' -f $Center),
  ('beta={0}' -f $Beta),
  ('mode={0}' -f $Mode),
  ('missing_score_behavior={0}' -f $MissingScoreBehavior),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('cmd_args={0}' -f ($argList -join ' '))
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached YOLOMG pred-label rescore.'
Get-Content $metaFile

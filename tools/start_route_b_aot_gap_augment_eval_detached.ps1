param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$SourceResultsFolder = 'papers\TransVisDrone\runs\val\AOT_URAP\fulltest_conf0p1_wport_baseline\aotpredictions',
  [string]$TrackletJsonl = 'artifacts\route_b_official\aot_fulltest_conf0p1_wport_baseline_tracklets\tracklets_with_multihead_scores_fulltest_conf0p1.jsonl',
  [string]$OutDir = 'artifacts\route_b_official\aot_fulltest_conf0p1_action_gap_augment',
  [string]$DatasetPath = 'D:\URAP_datasets\AOT\part1',
  [double]$DetectionThreshold = 0.2,
  [string]$ScoreField = 'video_action_model_fusion_score',
  [double]$MinScore = 0.9,
  [int]$MaxGap = 3,
  [int]$MinTrackletRows = 8,
  [double]$DuplicateIou = 0.5,
  [double]$ConfScale = 1.0,
  [string]$RunId = 'route_b_aot_gap_augment_eval',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\route_b_official\aot_fulltest_conf0p1_action_gap_augment_runner')
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -Path $SourceResultsFolder -PathType Container)) { throw "SourceResultsFolder not found: $SourceResultsFolder" }
if (-not (Test-Path -Path $TrackletJsonl -PathType Leaf)) { throw "TrackletJsonl not found: $TrackletJsonl" }
if (-not (Test-Path -Path $DatasetPath -PathType Container)) { throw "DatasetPath not found: $DatasetPath" }

if (-not [System.IO.Path]::IsPathRooted($SourceResultsFolder)) {
  $SourceResultsFolder = Join-Path $RepoRoot $SourceResultsFolder
}
if (-not [System.IO.Path]::IsPathRooted($TrackletJsonl)) {
  $TrackletJsonl = Join-Path $RepoRoot $TrackletJsonl
}
if (-not [System.IO.Path]::IsPathRooted($OutDir)) {
  $OutDir = Join-Path $RepoRoot $OutDir
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*aot_augment_prediction_parts_from_tracklet_gaps.py*') {
      Write-Host "AOT gap augment already running: pid=$existingPid"
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

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}_augment.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}_augment.err.txt" -f $RunId, $ts)
$evalRunnerDir = Join-Path $OutputRoot 'official_eval_runner'
$evalDir = Join-Path $OutDir 'official_eval'
$augmentedPredictions = Join-Path $OutDir 'aotpredictions'

$args = @(
  'tools\aot_augment_prediction_parts_from_tracklet_gaps.py',
  '--results-folder', $SourceResultsFolder,
  '--tracklet-jsonl', $TrackletJsonl,
  '--out-dir', $OutDir,
  '--score-field', $ScoreField,
  '--min-score', [string]$MinScore,
  '--max-gap', [string]$MaxGap,
  '--min-tracklet-rows', [string]$MinTrackletRows,
  '--duplicate-iou', [string]$DuplicateIou,
  '--conf-scale', [string]$ConfScale
)
$argumentString = ConvertTo-WindowsArgumentString -Values $args

$process = Start-Process `
  -FilePath $Python `
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
  ('python={0}' -f $Python),
  ('run_id={0}' -f $RunId),
  ('repo_root={0}' -f $RepoRoot),
  ('source_results_folder={0}' -f $SourceResultsFolder),
  ('tracklet_jsonl={0}' -f $TrackletJsonl),
  ('out_dir={0}' -f $OutDir),
  ('augmented_predictions={0}' -f $augmentedPredictions),
  ('evaluation_folder={0}' -f $evalDir),
  ('dataset_path={0}' -f $DatasetPath),
  ('detection_threshold={0}' -f $DetectionThreshold),
  ('score_field={0}' -f $ScoreField),
  ('min_score={0}' -f $MinScore),
  ('max_gap={0}' -f $MaxGap),
  ('min_tracklet_rows={0}' -f $MinTrackletRows),
  ('duplicate_iou={0}' -f $DuplicateIou),
  ('conf_scale={0}' -f $ConfScale),
  ('output_root={0}' -f $OutputRoot),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('eval_runner_dir={0}' -f $evalRunnerDir),
  ('cmd_args={0}' -f $argumentString)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached AOT gap augment.'
Get-Content $metaFile

Wait-Process -Id $process.Id
$process.Refresh()
$exitCode = $process.ExitCode
if ($null -eq $exitCode) {
  $exitCode = 0
}
if ($exitCode -ne 0) {
  Write-Host "AOT gap augment failed: exit_code=$exitCode"
  if (Test-Path $stdout) { Get-Content $stdout -Tail 80 }
  if (Test-Path $stderr) { Get-Content $stderr -Tail 80 }
  exit $exitCode
}

Write-Host 'AOT gap augment completed; launching official evaluation.'
& (Join-Path $PSScriptRoot 'start_route_b_aot_official_eval_detached.ps1') `
  -RepoRoot $RepoRoot `
  -ResultsFolder $augmentedPredictions `
  -EvaluationFolder $evalDir `
  -DatasetPath $DatasetPath `
  -DetectionThreshold $DetectionThreshold `
  -OutputRoot $evalRunnerDir `
  -RunId $RunId

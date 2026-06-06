param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$Weights,
  [string]$TrackletJsonl = 'artifacts\nps_sota_research\tvd_nps_test_tracklets_v2\proposal_tracklets.jsonl',
  [string]$FrameRoot = 'D:\URAP_datasets\TransVisDrone\NPS\AllFrames\test',
  [string]$PredictionsGtPkl = 'papers\TransVisDrone\runs\val\NPS_URAP_D\nps_test_best_aug_bs8_half\predictionsgt\predictionsgt_split_0.pkl',
  [string]$BaselineJson = 'artifacts\nps_sota_research\tvd_nps_test_recomputed_metrics.json',
  [string]$ScoreOut = 'artifacts\nps_sota_research\tvd_nps_test_tracklets_v2\vatd_scores_crop_full.jsonl',
  [string]$AttachedOut = 'artifacts\nps_sota_research\tvd_nps_test_tracklets_v2\tracklets_with_vatd_scores_crop_full.jsonl',
  [string]$SweepOutJson = 'artifacts\nps_sota_research\tvd_nps_test_action_sweep_crop_full.json',
  [string]$ComparisonOutCsv = 'artifacts\nps_sota_research\tvd_nps_test_action_sweep_crop_full_comparison.csv',
  [string]$ComparisonOutJson = 'artifacts\nps_sota_research\tvd_nps_test_action_sweep_crop_full_comparison.json',
  [string]$BestPkl = 'artifacts\nps_sota_research\tvd_nps_test_action_best_crop_full_predictionsgt.pkl',
  [double]$ErrorScale = 0.02,
  [int]$MinTrackletRows = 4,
  [int]$ScoreBatchSize = 512,
  [int]$NumWorkers = 0,
  [int]$FrameCacheSize = 256,
  [string]$Centers = '0.01 0.03 0.05 0.07 0.10 0.15 0.20 0.30 0.40 0.50',
  [string]$Betas = '0.005 0.01 0.02 0.05 0.10 0.20',
  [string[]]$Modes = @('boost-only', 'suppress-only', 'additive', 'gated-boost-low'),
  [string]$ScoreGates = '0.05 0.10 0.20',
  [string]$RunId = 'tvd_nps_vatd_score_sweep',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\nps_sota_research\tvd_nps_vatd_score_sweep_runner')
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not $Weights) { throw 'Weights must be provided' }
if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -Path $Weights -PathType Leaf)) { throw "Weights not found: $Weights" }
if (-not (Test-Path -Path $TrackletJsonl -PathType Leaf)) { throw "TrackletJsonl not found: $TrackletJsonl" }
if (-not (Test-Path -Path $FrameRoot -PathType Container)) { throw "FrameRoot not found: $FrameRoot" }
if (-not (Test-Path -Path $PredictionsGtPkl -PathType Leaf)) { throw "PredictionsGtPkl not found: $PredictionsGtPkl" }
if (-not (Test-Path -Path $BaselineJson -PathType Leaf)) { throw "BaselineJson not found: $BaselineJson" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*$RunId*") {
      Write-Host "NPS VATD score+sweep already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

foreach ($path in @($ScoreOut, $AttachedOut, $SweepOutJson, $ComparisonOutCsv, $ComparisonOutJson, $BestPkl)) {
  $parent = Split-Path -Parent $path
  if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logDir "runner_${RunId}_${ts}.err.txt"
$cmdFile = Join-Path $OutputRoot "$RunId.cmd.ps1"

$modeArgs = ($Modes | ForEach-Object { "'$_'" }) -join ', '
$script = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$RepoRoot'
`$env:PYTHONPATH = '$RepoRoot'
Write-Host '{"kind":"nps_vatd_score_sweep_start","stage":"score"}'
& '$Python' -m qstr_dronedet.cli score-vatd-motion-action-tracklets --tracklet-jsonl '$TrackletJsonl' --weights '$Weights' --out '$ScoreOut' --frame-root '$FrameRoot' --error-scale '$ErrorScale' --min-tracklet-rows '$MinTrackletRows' --batch-size '$ScoreBatchSize' --num-workers '$NumWorkers' --frame-cache-size '$FrameCacheSize' --fusion-mode motion_action
Write-Host '{"kind":"nps_vatd_score_sweep_progress","stage":"attach"}'
& '$Python' -m qstr_dronedet.cli attach-vatd-scores-to-tracklets --tracklet-jsonl '$TrackletJsonl' --vatd-scores '$ScoreOut' --out '$AttachedOut'
Write-Host '{"kind":"nps_vatd_score_sweep_progress","stage":"sweep"}'
& '$Python' tools\sweep_tvd_predictionsgt_action_rescore.py --tvd-root papers\TransVisDrone --predictionsgt-pkl '$PredictionsGtPkl' --tracklet-jsonl '$AttachedOut' --score-field vatd_score --centers '$Centers' --betas '$Betas' --modes $($Modes -join ' ') --score-gates '$ScoreGates' --missing-score-behaviors keep --out-json '$SweepOutJson' --write-best-pkl '$BestPkl'
Write-Host '{"kind":"nps_vatd_score_sweep_progress","stage":"claim_gate"}'
& '$Python' tools\collect_vatd_nps_sweep_results.py --sweep-json '$SweepOutJson' --baseline-json '$BaselineJson' --out-csv '$ComparisonOutCsv' --out-json '$ComparisonOutJson'
Write-Host '{"kind":"nps_vatd_score_sweep_done","stage":"done","sweep_out":"$SweepOutJson","comparison_json":"$ComparisonOutJson"}'
"@
$script | Set-Content -Path $cmdFile -Encoding utf8

$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $cmdFile) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "run_id=$RunId",
  "repo_root=$RepoRoot",
  "python=$Python",
  "weights=$Weights",
  "tracklet_jsonl=$TrackletJsonl",
  "frame_root=$FrameRoot",
  "predictionsgt_pkl=$PredictionsGtPkl",
  "baseline_json=$BaselineJson",
  "score_out=$ScoreOut",
  "attached_out=$AttachedOut",
  "sweep_out_json=$SweepOutJson",
  "comparison_out_csv=$ComparisonOutCsv",
  "comparison_out_json=$ComparisonOutJson",
  "best_pkl=$BestPkl",
  "output_root=$OutputRoot",
  "stdout=$stdout",
  "stderr=$stderr",
  "cmd_file=$cmdFile"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host 'Started detached NPS VATD score+sweep pipeline.'
Get-Content $metaFile

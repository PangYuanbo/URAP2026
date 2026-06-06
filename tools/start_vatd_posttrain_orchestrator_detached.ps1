param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId = 'vatd_posttrain_orchestrator_20260605',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\vatd_posttrain_orchestrator_20260605'),
  [int]$PollSeconds = 60,
  [string]$AotWeights = 'artifacts\yolomg_action\vatd_motion_action_train_full_e1_b1024_crop64_nw0_nopin_20260605\vatd_motion_action.pt',
  [string]$NpsWeights = 'artifacts\nps_sota_research\tvd_nps_val_vatd_train_crop_full_20260605\vatd_motion_action.pt'
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"
$runnerFile = Join-Path $OutputRoot "$RunId.runner.ps1"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*$RunId.runner.ps1*") {
      Write-Host "VATD posttrain orchestrator already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logsDir "runner_${RunId}_${ts}.err.txt"

$runner = @"
param()
`$ErrorActionPreference = 'Stop'
Set-Location '$RepoRoot'

`$pollSeconds = $PollSeconds
`$aotWeights = '$AotWeights'
`$npsWeights = '$NpsWeights'
`$outputRoot = '$OutputRoot'
`$aotScoreMarker = Join-Path `$outputRoot 'aot_score_launched.marker'
`$aotEvalMarker = Join-Path `$outputRoot 'aot_official_launched.marker'
`$npsMarker = Join-Path `$outputRoot 'nps_score_sweep_launched.marker'

`$aotScores = 'artifacts\route_b_official\aot_fulltest_vatd_motion_action_score_e1_shuffle_20260605\vatd_scores.jsonl'
`$aotScoreRunner = 'artifacts\route_b_official\aot_fulltest_vatd_motion_action_score_e1_shuffle_runner_20260605'
`$aotScoreRunId = 'aot_fulltest_vatd_motion_action_score_e1_shuffle_20260605'
`$aotOfficialRoot = 'artifacts\route_b_official\aot_fulltest_vatd_e1_shuffle_suppress_c0p10_b0p20_official_runner_20260605'
`$aotOfficialOut = 'artifacts\route_b_official\aot_fulltest_vatd_e1_shuffle_suppress_c0p10_b0p20_official_20260605'
`$npsRunner = 'artifacts\nps_sota_research\tvd_nps_test_vatd_score_sweep_crop_full_runner_20260605'

function Write-Event([string]`$Kind, [hashtable]`$Extra) {
  `$row = [ordered]@{ kind = `$Kind; time = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') }
  foreach (`$key in `$Extra.Keys) { `$row[`$key] = `$Extra[`$key] }
  Write-Host (`$row | ConvertTo-Json -Compress)
}

function Test-LauncherRunning([string]`$OutputRootPath, [string]`$RunIdValue, [string]`$CommandPattern = '') {
  `$pidPath = Join-Path `$OutputRootPath "`$RunIdValue.pid"
  if (Test-Path `$pidPath) {
    `$pidValue = Get-Content `$pidPath -ErrorAction SilentlyContinue | Select-Object -First 1
    if (`$pidValue) {
      `$proc = Get-CimInstance Win32_Process -Filter "ProcessId = `$pidValue" -ErrorAction SilentlyContinue
      if (`$proc) { return `$true }
      `$children = Get-CimInstance Win32_Process -Filter "ParentProcessId = `$pidValue" -ErrorAction SilentlyContinue
      if (`$children) { return `$true }
    }
  }
  if (`$CommandPattern) {
    `$matching = Get-CimInstance Win32_Process | Where-Object { `$_.CommandLine -like `$CommandPattern } | Select-Object -First 1
    if (`$matching) { return `$true }
  }
  return `$false
}

Write-Event 'vatd_posttrain_orchestrator_start' @{ poll_seconds = `$pollSeconds; aot_weights = `$aotWeights; nps_weights = `$npsWeights }

while (`$true) {
  if ((Test-Path `$aotWeights) -and -not (Test-Path `$aotScoreMarker)) {
    Write-Event 'launch_aot_score' @{ weights = `$aotWeights; scores = `$aotScores }
    & tools\start_route_b_vatd_motion_action_score_detached.ps1 -TrackletJsonl 'artifacts\route_b_official\aot_fulltest_wport_baseline_tracklets\route_b_tracklets_min2_gap2_with_paths.jsonl' -Weights `$aotWeights -Out `$aotScores -FrameRoot 'D:\URAP_datasets\AOT\part1' -ErrorScale 0.02 -MinTrackletRows 4 -BatchSize 512 -NumWorkers 0 -FrameCacheSize 256 -FusionMode motion_action -RunId `$aotScoreRunId -OutputRoot `$aotScoreRunner
    "launched=`$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Set-Content -Path `$aotScoreMarker -Encoding ascii
  }

  if ((Test-Path `$aotScores) -and -not (Test-LauncherRunning `$aotScoreRunner `$aotScoreRunId '*score-vatd-motion-action-tracklets*') -and -not (Test-Path `$aotEvalMarker)) {
    Write-Event 'launch_aot_official' @{ scores = `$aotScores; out_dir = `$aotOfficialOut }
    & tools\start_route_b_vatd_aot_official_pipeline_detached.ps1 -VatdScores `$aotScores -OutDir `$aotOfficialOut -Center 0.10 -Beta 0.20 -Mode suppress-only -DetectionThreshold 0.2 -RunId 'aot_fulltest_vatd_e1_shuffle_suppress_c0p10_b0p20_official_20260605' -OutputRoot `$aotOfficialRoot
    "launched=`$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Set-Content -Path `$aotEvalMarker -Encoding ascii
  }

  if ((Test-Path `$npsWeights) -and -not (Test-Path `$npsMarker)) {
    Write-Event 'launch_nps_score_sweep' @{ weights = `$npsWeights; runner = `$npsRunner }
    & tools\start_nps_vatd_score_sweep_detached.ps1 -Weights `$npsWeights -ScoreOut 'artifacts\nps_sota_research\tvd_nps_test_tracklets_v2\vatd_scores_crop_full.jsonl' -AttachedOut 'artifacts\nps_sota_research\tvd_nps_test_tracklets_v2\tracklets_with_vatd_scores_crop_full.jsonl' -SweepOutJson 'artifacts\nps_sota_research\tvd_nps_test_action_sweep_crop_full.json' -BestPkl 'artifacts\nps_sota_research\tvd_nps_test_action_best_crop_full_predictionsgt.pkl' -RunId 'tvd_nps_test_vatd_score_sweep_crop_full_20260605' -OutputRoot `$npsRunner
    "launched=`$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Set-Content -Path `$npsMarker -Encoding ascii
  }

  if ((Test-Path `$aotEvalMarker) -and (Test-Path `$npsMarker)) {
    Write-Event 'vatd_posttrain_orchestrator_done' @{ aot_eval_marker = `$aotEvalMarker; nps_marker = `$npsMarker }
    break
  }

  Write-Event 'vatd_posttrain_orchestrator_wait' @{
    aot_weights_ready = (Test-Path `$aotWeights)
    aot_scores_ready = (Test-Path `$aotScores)
    nps_weights_ready = (Test-Path `$npsWeights)
  }
  Start-Sleep -Seconds `$pollSeconds
}
"@

$runner | Set-Content -Path $runnerFile -Encoding utf8
$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runnerFile) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii

@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "run_id=$RunId",
  "repo_root=$RepoRoot",
  "output_root=$OutputRoot",
  "poll_seconds=$PollSeconds",
  "aot_weights=$AotWeights",
  "nps_weights=$NpsWeights",
  "runner_file=$runnerFile",
  "stdout=$stdout",
  "stderr=$stderr"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host 'Started detached VATD posttrain orchestrator.'
Get-Content $metaFile

param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$Weights,
  [string]$FrameRoot = 'U:\URAP_datasets\TransVisDrone\NPS\AllFrames\test',
  [string]$ScoreRunId = 'nps_ego_adaptive_vatd_score_sweep',
  [string]$ScoreOutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\nps_sota_research\nps_ego_adaptive_vatd_score_sweep_runner'),
  [string]$ComparisonJson = 'artifacts\nps_sota_research\tvd_nps_test_ego_adaptive_vatd_sweep_comparison.json',
  [string]$Target20Json = 'artifacts\nps_sota_research\tvd_nps_test_ego_adaptive_vatd_20pct_target_gate.json',
  [string]$PrimaryMetric = 'map50',
  [string]$Direction = 'higher',
  [double]$MinRelativeImprovement = 0.20,
  [string[]]$Guards = @('recall:higher', 'map5095:higher'),
  [int]$PollSeconds = 60,
  [int]$TimeoutMinutes = 0,
  [string]$RunId = 'ego_adaptive_nps_posttrain_watcher',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\ego_adaptive_vatd\nps_posttrain_watcher')
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not $Weights) { throw 'Weights is required.' }
if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -Path $FrameRoot -PathType Container)) { throw "FrameRoot not found: $FrameRoot" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"
$runnerFile = Join-Path $OutputRoot "$RunId.runner.ps1"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*$RunId.runner.ps1*") {
      Write-Host "Ego-Adaptive NPS posttrain watcher already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logDir "runner_${RunId}_${ts}.err.txt"
$guardArgs = ($Guards | ForEach-Object { "--guard '$($_)'" }) -join ' '

$script = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$RepoRoot'
`$env:PYTHONPATH = '$RepoRoot'
`$start = Get-Date
Write-Host "{`"kind`":`"ego_adaptive_nps_posttrain_start`",`"stage`":`"wait_weights`",`"weights`":`"$Weights`"}"
while (-not (Test-Path '$Weights')) {
  if ($TimeoutMinutes -gt 0 -and ((Get-Date) - `$start).TotalMinutes -gt $TimeoutMinutes) { throw 'timeout waiting for weights: $Weights' }
  Start-Sleep -Seconds $PollSeconds
}
Write-Host "{`"kind`":`"ego_adaptive_nps_posttrain_progress`",`"stage`":`"weights_ready`",`"weights`":`"$Weights`"}"
& '$RepoRoot\tools\start_nps_ego_adaptive_vatd_score_sweep_detached.ps1' -RepoRoot '$RepoRoot' -Python '$Python' -Weights '$Weights' -FrameRoot '$FrameRoot' -RunId '$ScoreRunId' -OutputRoot '$ScoreOutputRoot'
Write-Host "{`"kind`":`"ego_adaptive_nps_posttrain_progress`",`"stage`":`"score_sweep_launched`",`"comparison_json`":`"$ComparisonJson`"}"
while (-not (Test-Path '$ComparisonJson')) {
  if ($TimeoutMinutes -gt 0 -and ((Get-Date) - `$start).TotalMinutes -gt $TimeoutMinutes) { throw 'timeout waiting for comparison json: $ComparisonJson' }
  Start-Sleep -Seconds $PollSeconds
}
Write-Host "{`"kind`":`"ego_adaptive_nps_posttrain_progress`",`"stage`":`"comparison_ready`",`"comparison_json`":`"$ComparisonJson`"}"
& '$Python' tools\collect_ego_adaptive_vatd_20pct_targets.py --comparison-json '$ComparisonJson' --out-json '$Target20Json' --method-name 'Ego-Adaptive VATD' --baseline-name 'TransVisDrone' --primary-metric '$PrimaryMetric' --direction '$Direction' --min-relative-improvement '$MinRelativeImprovement' $guardArgs
Write-Host "{`"kind`":`"ego_adaptive_nps_posttrain_done`",`"stage`":`"target20_gate`",`"target20_json`":`"$Target20Json`"}"
"@
$script | Set-Content -Path $runnerFile -Encoding utf8

$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $runnerFile) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "run_id=$RunId",
  "repo_root=$RepoRoot",
  "python=$Python",
  "weights=$Weights",
  "frame_root=$FrameRoot",
  "score_run_id=$ScoreRunId",
  "score_output_root=$ScoreOutputRoot",
  "comparison_json=$ComparisonJson",
  "target20_json=$Target20Json",
  "primary_metric=$PrimaryMetric",
  "direction=$Direction",
  "min_relative_improvement=$MinRelativeImprovement",
  "guards=$($Guards -join ',')",
  "poll_seconds=$PollSeconds",
  "timeout_minutes=$TimeoutMinutes",
  "runner_file=$runnerFile",
  "stdout=$stdout",
  "stderr=$stderr"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host 'Started detached Ego-Adaptive NPS posttrain watcher.'
Get-Content $metaFile

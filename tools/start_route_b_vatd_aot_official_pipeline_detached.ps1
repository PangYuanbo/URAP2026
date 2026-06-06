param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$SourceResultsFolder = 'papers\TransVisDrone\runs\val\AOT_URAP\fulltest_conf0p2_wport_baseline\aotpredictions',
  [string]$TrackletJsonl = 'artifacts\route_b_official\aot_fulltest_wport_baseline_tracklets\route_b_tracklets_min2_gap2_with_paths.jsonl',
  [string]$VatdScores,
  [string]$OutDir = 'artifacts\route_b_official\aot_fulltest_vatd_motion_action_official',
  [string]$DatasetPath = 'D:\URAP_datasets\AOT\part1',
  [double]$DetectionThreshold = 0.2,
  [string]$ScoreField = 'vatd_score',
  [double]$Center = 0.5,
  [double]$Beta = 0.4,
  [string]$Mode = 'suppress-only',
  [double]$ProtectRawScoreAt = -1.0,
  [int]$MinTrackletRows = 1,
  [string]$RunId = 'route_b_vatd_aot_official',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\route_b_official\aot_fulltest_vatd_motion_action_official_runner')
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -Path $SourceResultsFolder -PathType Container)) { throw "SourceResultsFolder not found: $SourceResultsFolder" }
if (-not (Test-Path -Path $TrackletJsonl -PathType Leaf)) { throw "TrackletJsonl not found: $TrackletJsonl" }
if (-not $VatdScores) { throw 'VatdScores must be provided' }
if (-not (Test-Path -Path $VatdScores -PathType Leaf)) { throw "VatdScores not found: $VatdScores" }
if (-not (Test-Path -Path $DatasetPath -PathType Container)) { throw "DatasetPath not found: $DatasetPath" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

$attachedTracklets = Join-Path $OutDir 'tracklets_with_vatd_scores.jsonl'
$rescoreDir = Join-Path $OutDir 'rescored'
$evalDir = Join-Path $OutDir 'official_eval'
$evalRunnerDir = Join-Path $OutputRoot 'official_eval_runner'
$claimGateRunnerDir = Join-Path $OutputRoot 'claim_gate_runner'
$claimComparisonCsv = Join-Path $OutDir 'aot_official_claim_comparison.csv'
$claimComparisonJson = Join-Path $OutDir 'aot_official_claim_comparison.json'
$claimSummaryGlob = Join-Path (Join-Path $evalDir 'summaries') 'result_metrics*_summary*.json'

function Run-Step([string]$Name, [string[]]$StepArgs, [string]$Stdout, [string]$Stderr) {
  Write-Host "Running $Name"
  $cleanArgs = @($StepArgs | Where-Object { $null -ne $_ -and [string]$_ -ne '' })
  @(
    "step=$Name",
    "python=$Python",
    "working_directory=$RepoRoot",
    "args=$($cleanArgs -join ' ')"
  ) | Set-Content -Path ($Stdout + '.cmd.txt') -Encoding utf8
  $proc = Start-Process -FilePath $Python -ArgumentList $cleanArgs -WorkingDirectory $RepoRoot -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -Wait -PassThru -WindowStyle Hidden
  if ($proc.ExitCode -ne 0) {
    if (Test-Path $Stdout) { Get-Content $Stdout -Tail 60 }
    if (Test-Path $Stderr) { Get-Content $Stderr -Tail 60 }
    throw "$Name failed with exit code $($proc.ExitCode)"
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$attachOut = Join-Path $logsDir "runner_${RunId}_${ts}_attach.out.txt"
$attachErr = Join-Path $logsDir "runner_${RunId}_${ts}_attach.err.txt"
$rescoreOut = Join-Path $logsDir "runner_${RunId}_${ts}_rescore.out.txt"
$rescoreErr = Join-Path $logsDir "runner_${RunId}_${ts}_rescore.err.txt"

Run-Step 'VATD score attach' @(
  '-m', 'qstr_dronedet.cli', 'attach-vatd-scores-to-tracklets',
  '--tracklet-jsonl', $TrackletJsonl,
  '--vatd-scores', $VatdScores,
  '--out', $attachedTracklets,
  '--score-field', $ScoreField
) $attachOut $attachErr

$rescoreArgs = @(
  '-m', 'qstr_dronedet.cli', 'rescore-aot-prediction-parts-by-tracklets',
  '--results-folder', $SourceResultsFolder,
  '--tracklet-jsonl', $attachedTracklets,
  '--out-dir', $rescoreDir,
  '--score-field', $ScoreField,
  '--center', [string]$Center,
  '--beta', [string]$Beta,
  '--mode', $Mode,
  '--min-tracklet-rows', [string]$MinTrackletRows,
  '--missing-score-behavior', 'keep'
)
if ($ProtectRawScoreAt -ge 0.0) {
  $rescoreArgs += @('--protect-raw-score-at', [string]$ProtectRawScoreAt)
}
Run-Step 'VATD AOT pkl rescore' $rescoreArgs $rescoreOut $rescoreErr

@(
  ('started={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
  ('run_id={0}' -f $RunId),
  ('repo_root={0}' -f $RepoRoot),
  ('source_results_folder={0}' -f $SourceResultsFolder),
  ('tracklet_jsonl={0}' -f $TrackletJsonl),
  ('vatd_scores={0}' -f $VatdScores),
  ('attached_tracklets={0}' -f $attachedTracklets),
  ('rescore_dir={0}' -f $rescoreDir),
  ('rescored_aotpredictions={0}' -f (Join-Path $rescoreDir 'aotpredictions')),
  ('evaluation_folder={0}' -f $evalDir),
  ('claim_summary_glob={0}' -f $claimSummaryGlob),
  ('claim_comparison_csv={0}' -f $claimComparisonCsv),
  ('claim_comparison_json={0}' -f $claimComparisonJson),
  ('claim_gate_runner_dir={0}' -f $claimGateRunnerDir),
  ('dataset_path={0}' -f $DatasetPath),
  ('detection_threshold={0}' -f $DetectionThreshold),
  ('score_field={0}' -f $ScoreField),
  ('center={0}' -f $Center),
  ('beta={0}' -f $Beta),
  ('mode={0}' -f $Mode),
  ('protect_raw_score_at={0}' -f $ProtectRawScoreAt),
  ('output_root={0}' -f $OutputRoot),
  ('eval_runner_dir={0}' -f $evalRunnerDir),
  ('attach_stdout={0}' -f $attachOut),
  ('attach_stderr={0}' -f $attachErr),
  ('rescore_stdout={0}' -f $rescoreOut),
  ('rescore_stderr={0}' -f $rescoreErr)
) | Set-Content -Encoding utf8 -Path $metaFile

Write-Host 'Launching detached official AOT evaluation for VATD-rescored predictions.'
& (Join-Path $PSScriptRoot 'start_route_b_aot_official_eval_detached.ps1') `
  -RepoRoot $RepoRoot `
  -ResultsFolder (Join-Path $rescoreDir 'aotpredictions') `
  -EvaluationFolder $evalDir `
  -DatasetPath $DatasetPath `
  -DetectionThreshold $DetectionThreshold `
  -OutputRoot $evalRunnerDir `
  -RunId $RunId

Write-Host 'Launching detached AOT official claim gate watcher.'
& (Join-Path $PSScriptRoot 'start_route_b_aot_official_claim_gate_detached.ps1') `
  -RepoRoot $RepoRoot `
  -Python $Python `
  -SummaryGlob $claimSummaryGlob `
  -OutCsv $claimComparisonCsv `
  -OutJson $claimComparisonJson `
  -OutputRoot $claimGateRunnerDir `
  -RunId $RunId

Write-Host 'Started VATD AOT official pipeline.'
Get-Content $metaFile

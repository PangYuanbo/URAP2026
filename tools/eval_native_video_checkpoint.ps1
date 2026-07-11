param(
  [string]$RunId,
  [string]$Weights = "",
  [ValidateSet("val", "test")]
  [string]$Split = "val",
  [string]$DataRoot = "U:\URAP_datasets",
  [int]$BatchSize = 16,
  [int]$MaxSamples = 0,
  [double]$ScoreThreshold = 0.0,
  [int]$TopK = 32,
  [int]$ProposalPrefilterTopK = 0,
  [double]$ProposalScoreWeight = 0.0,
  [double]$QualityScoreWeight = 0.0,
  [double]$NmsIouThreshold = 0.5,
  [int]$SamuraiMotionRerank = 0,
  [double]$SamuraiAppearanceWeight = 0.60,
  [double]$SamuraiMotionIouWeight = 0.30,
  [double]$SamuraiCenterWeight = 0.05,
  [double]$SamuraiCenterSigmaPixels = 96.0,
  [double]$SamuraiUpdateScoreThreshold = 0.05,
  [double]$SamuraiUpdateMotionIouThreshold = 0.0,
  [int]$SamuraiLostTau = 8,
  [double]$SamuraiVelocityMomentum = 0.6,
  [int]$SamuraiTrackletRerank = 0,
  [int]$SamuraiTrackletCandidateTopK = 32,
  [double]$SamuraiTrackletMatchThreshold = 0.15,
  [int]$SamuraiTrackletMaxGap = 2,
  [double]$SamuraiTrackletSpawnScoreThreshold = 0.02,
  [double]$SamuraiTrackletLengthNorm = 4.0,
  [double]$SamuraiTrackletAppearanceWeight = 0.65,
  [double]$SamuraiTrackletWeight = 0.35,
  [double]$SamuraiTrackletUnmatchedScale = 0.5,
  [int]$ActionChunkBackfill = 0,
  [int]$ActionChunkMaxStep = 0,
  [int]$ActionChunkTopK = 0,
  [double]$ActionChunkScoreDecay = 0.85,
  [ValidateSet("add", "support")]
  [string]$ActionChunkMergeMode = "add",
  [double]$ActionChunkSupportIou = 0.3,
  [double]$ActionChunkSupportWeight = 0.4,
  [int]$ActionChunkKeepUnmatched = 0,
  [int]$ExportLogEvery = 50,
  [double[]]$SweepScoreThresholds = @(0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5),
  [int[]]$SweepTopKs = @(1, 2, 4, 8, 16, 32),
  [string]$PrimaryMetric = "map50",
  [double]$MinDelta = 0.0,
  [int]$RequireFullSplitBaseline = 0,
  [string]$EvalName = "",
  [string]$CacheDir = "",
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Device = "auto",
  [int]$NoEma = 0
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"

if ([string]::IsNullOrWhiteSpace($RunId) -and [string]::IsNullOrWhiteSpace($Weights)) {
  throw "Pass -RunId or -Weights."
}

if ([string]::IsNullOrWhiteSpace($Weights)) {
  $RunDir = Join-Path $Repo "artifacts\native_video_detector\$RunId"
  $Weights = Join-Path $RunDir "native_video_detector_latest.pt"
  if (-not (Test-Path $Weights)) {
    $Weights = Join-Path $RunDir "native_video_detector.pt"
  }
} else {
  $RunDir = Split-Path -Parent $Weights
}

if (-not (Test-Path $Python)) {
  throw "Python venv not found: $Python"
}
if (-not (Test-Path $Weights)) {
  throw "Weights not found: $Weights"
}

$FramesDir = Join-Path $DataRoot "TransVisDrone\NPS\AllFrames\$Split"
if ($Split -eq "val") {
  $GtCsv = Join-Path $Repo "artifacts\nps_sota_research\tvd_nps_val_route_b_v2\gt.csv"
} else {
  $GtCsv = Join-Path $Repo "artifacts\nps_sota_research\tvd_nps_test_route_b_v2\gt.csv"
}
if (-not (Test-Path $FramesDir)) {
  throw "Frames dir not found: $FramesDir"
}
if (-not (Test-Path $GtCsv)) {
  throw "GT CSV not found: $GtCsv"
}

if ([string]::IsNullOrWhiteSpace($EvalName)) {
  $EvalName = "eval_$Split"
}
$EvalDir = Join-Path $RunDir $EvalName
New-Item -ItemType Directory -Force -Path $EvalDir | Out-Null
$Pkl = Join-Path $EvalDir "predictionsgt.pkl"
$EvalJson = Join-Path $EvalDir "eval.json"
$SweepJson = Join-Path $EvalDir "threshold_sweep.json"
$SweepCsv = Join-Path $EvalDir "threshold_sweep.csv"
$BestEvalJson = Join-Path $EvalDir "best_eval.json"
$CompareJson = Join-Path $EvalDir "baseline_comparison.json"

$ExportArgs = @(
  "tools\export_native_video_predictionsgt.py",
  "--weights", $Weights,
  "--frames-dir", $FramesDir,
  "--gt-csv", $GtCsv,
  "--out-pkl", $Pkl,
  "--batch-size", "$BatchSize",
  "--score-threshold", "$ScoreThreshold",
  "--top-k", "$TopK",
  "--proposal-prefilter-topk", "$ProposalPrefilterTopK",
  "--proposal-score-weight", "$ProposalScoreWeight",
  "--quality-score-weight", "$QualityScoreWeight",
  "--nms-iou-threshold", "$NmsIouThreshold",
  "--samurai-appearance-weight", "$SamuraiAppearanceWeight",
  "--samurai-motion-iou-weight", "$SamuraiMotionIouWeight",
  "--samurai-center-weight", "$SamuraiCenterWeight",
  "--samurai-center-sigma-pixels", "$SamuraiCenterSigmaPixels",
  "--samurai-update-score-threshold", "$SamuraiUpdateScoreThreshold",
  "--samurai-update-motion-iou-threshold", "$SamuraiUpdateMotionIouThreshold",
  "--samurai-lost-tau", "$SamuraiLostTau",
  "--samurai-velocity-momentum", "$SamuraiVelocityMomentum",
  "--samurai-tracklet-candidate-topk", "$SamuraiTrackletCandidateTopK",
  "--samurai-tracklet-match-threshold", "$SamuraiTrackletMatchThreshold",
  "--samurai-tracklet-max-gap", "$SamuraiTrackletMaxGap",
  "--samurai-tracklet-spawn-score-threshold", "$SamuraiTrackletSpawnScoreThreshold",
  "--samurai-tracklet-length-norm", "$SamuraiTrackletLengthNorm",
  "--samurai-tracklet-appearance-weight", "$SamuraiTrackletAppearanceWeight",
  "--samurai-tracklet-weight", "$SamuraiTrackletWeight",
  "--samurai-tracklet-unmatched-scale", "$SamuraiTrackletUnmatchedScale",
  "--action-chunk-max-step", "$ActionChunkMaxStep",
  "--action-chunk-top-k", "$ActionChunkTopK",
  "--action-chunk-score-decay", "$ActionChunkScoreDecay",
  "--action-chunk-merge-mode", "$ActionChunkMergeMode",
  "--action-chunk-support-iou", "$ActionChunkSupportIou",
  "--action-chunk-support-weight", "$ActionChunkSupportWeight",
  "--log-every", "$ExportLogEvery",
  "--device", "$Device"
)
if ($SamuraiMotionRerank -ne 0) {
  $ExportArgs += "--samurai-motion-rerank"
}
if ($SamuraiTrackletRerank -ne 0) {
  $ExportArgs += "--samurai-tracklet-rerank"
}
if ($ActionChunkBackfill -ne 0) {
  $ExportArgs += "--action-chunk-backfill"
}
if ($ActionChunkKeepUnmatched -ne 0) {
  $ExportArgs += "--action-chunk-keep-unmatched"
}
if ($NoEma -ne 0) {
  $ExportArgs += "--no-ema"
}
if (-not [string]::IsNullOrWhiteSpace($CacheDir)) {
  $ExportArgs += @("--cache-dir", $CacheDir)
}
if ($MaxSamples -gt 0) {
  $ExportArgs += @("--max-samples", "$MaxSamples")
}

& $Python $ExportArgs
if ($LASTEXITCODE -ne 0) {
  throw "Export failed with exit code $LASTEXITCODE"
}

& $Python "tools\eval_tvd_predictionsgt_pkl.py" "--predictionsgt-pkl" $Pkl "--out-json" $EvalJson
if ($LASTEXITCODE -ne 0) {
  throw "Evaluation failed with exit code $LASTEXITCODE"
}

$SweepArgs = @(
  "tools\sweep_tvd_predictionsgt_thresholds.py",
  "--predictionsgt-pkl", $Pkl,
  "--out-json", $SweepJson,
  "--out-csv", $SweepCsv,
  "--primary-metric", $PrimaryMetric,
  "--score-thresholds"
)
$SweepArgs += $SweepScoreThresholds | ForEach-Object { "$_" }
$SweepArgs += "--top-ks"
$SweepArgs += $SweepTopKs | ForEach-Object { "$_" }

& $Python $SweepArgs
if ($LASTEXITCODE -ne 0) {
  throw "Threshold sweep failed with exit code $LASTEXITCODE"
}

$SweepData = Get-Content -LiteralPath $SweepJson -Raw | ConvertFrom-Json
$BestEval = [ordered]@{
  predictionsgt_pkl = $Pkl
  images = [int]$SweepData.best.images
  labels = [int]$SweepData.best.labels
  detections = [int]$SweepData.best.detections
  precision = [double]$SweepData.best.precision
  recall = [double]$SweepData.best.recall
  map50 = [double]$SweepData.best.map50
  map5095 = [double]$SweepData.best.map5095
  f1 = [double]$SweepData.best.f1
  score_threshold = [double]$SweepData.best.score_threshold
  top_k = [int]$SweepData.best.top_k
  proposal_prefilter_top_k = $ProposalPrefilterTopK
  proposal_score_weight = $ProposalScoreWeight
  quality_score_weight = $QualityScoreWeight
  samurai_motion_rerank = [bool]($SamuraiMotionRerank -ne 0)
  samurai_appearance_weight = $SamuraiAppearanceWeight
  samurai_motion_iou_weight = $SamuraiMotionIouWeight
  samurai_center_weight = $SamuraiCenterWeight
  samurai_center_sigma_pixels = $SamuraiCenterSigmaPixels
  samurai_update_score_threshold = $SamuraiUpdateScoreThreshold
  samurai_update_motion_iou_threshold = $SamuraiUpdateMotionIouThreshold
  samurai_lost_tau = $SamuraiLostTau
  samurai_velocity_momentum = $SamuraiVelocityMomentum
  samurai_tracklet_rerank = [bool]($SamuraiTrackletRerank -ne 0)
  samurai_tracklet_candidate_topk = $SamuraiTrackletCandidateTopK
  samurai_tracklet_match_threshold = $SamuraiTrackletMatchThreshold
  samurai_tracklet_max_gap = $SamuraiTrackletMaxGap
  samurai_tracklet_spawn_score_threshold = $SamuraiTrackletSpawnScoreThreshold
  samurai_tracklet_length_norm = $SamuraiTrackletLengthNorm
  samurai_tracklet_appearance_weight = $SamuraiTrackletAppearanceWeight
  samurai_tracklet_weight = $SamuraiTrackletWeight
  samurai_tracklet_unmatched_scale = $SamuraiTrackletUnmatchedScale
  action_chunk_backfill = [bool]($ActionChunkBackfill -ne 0)
  action_chunk_max_step = $ActionChunkMaxStep
  action_chunk_top_k = $ActionChunkTopK
  action_chunk_score_decay = $ActionChunkScoreDecay
  action_chunk_merge_mode = $ActionChunkMergeMode
  action_chunk_support_iou = $ActionChunkSupportIou
  action_chunk_support_weight = $ActionChunkSupportWeight
  action_chunk_keep_unmatched = [bool]($ActionChunkKeepUnmatched -ne 0)
  max_samples = $MaxSamples
  full_split = [bool]($MaxSamples -le 0)
  threshold_sweep_json = $SweepJson
  threshold_sweep_csv = $SweepCsv
  eval_name = $EvalName
  weights = $Weights
  device = $Device
  no_ema = [bool]($NoEma -ne 0)
}
$BestEval | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $BestEvalJson -Encoding UTF8

$CompareArgs = @(
  "tools\compare_native_video_eval_baseline.py",
  "--eval-json", $BestEvalJson,
  "--out-json", $CompareJson,
  "--primary-metric", $PrimaryMetric,
  "--min-delta", "$MinDelta"
)
if ($RequireFullSplitBaseline -ne 0) {
  $CompareArgs += "--require-full-split"
}
& $Python $CompareArgs
$CompareExitCode = $LASTEXITCODE

Write-Output "Native video checkpoint evaluation complete"
Write-Output "RunDir: $RunDir"
Write-Output "Weights: $Weights"
Write-Output "Split: $Split"
Write-Output "PredictionsGT: $Pkl"
Write-Output "EvalJson: $EvalJson"
Write-Output "ThresholdSweepJson: $SweepJson"
Write-Output "ThresholdSweepCsv: $SweepCsv"
Write-Output "BestEvalJson: $BestEvalJson"
Write-Output "BaselineComparison: $CompareJson"
if ($CompareExitCode -ne 0) {
  Write-Output "BaselineComparisonStatus: below_baseline"
} else {
  Write-Output "BaselineComparisonStatus: beat_baseline"
}

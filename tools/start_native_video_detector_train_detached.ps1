param(
  [string]$RunId = "nps_native_video_mvp_b32_w0_u_amp_ckpt",
  [int]$Epochs = 30,
  [int]$BatchSize = 32,
  [int]$ImageSize = 320,
  [int]$ClipLen = 8,
  [int]$FutureLen = 4,
  [int]$NumQueries = 32,
  [int]$DModel = 192,
  [int]$Nhead = 4,
  [int]$EncoderLayers = 4,
  [int]$DecoderLayers = 2,
  [string]$EncoderMode = "factorized",
  [int]$PatchStride = 8,
  [int]$SpatialRefineLayers = 0,
  [int]$SpatialRefineKernel = 7,
  [double]$SpatialRefineExpansion = 2.0,
  [bool]$MotionChannels = $false,
  [ValidateSet("last", "samurai")]
  [string]$MemoryMode = "last",
  [double]$BoxSizeScale = 1.0,
  [ValidateSet("learned", "dense")]
  [string]$QueryMode = "learned",
  [double]$AnchorOffsetCells = 4.0,
  [ValidateSet("token", "conv")]
  [string]$DenseObjSource = "token",
  [ValidateSet("none", "pooled_cross")]
  [string]$MemoryAttention = "none",
  [int]$MemorySlots = 64,
  [ValidateSet("none", "slot_dot")]
  [string]$MemoryMatchMode = "none",
  [double]$MemoryMatchWeight = 0.0,
  [double]$MemoryMatchTemperature = 5.0,
  [ValidateSet("none", "samurai")]
  [string]$MotionScoreMode = "none",
  [double]$MotionScoreWeight = 1.0,
  [ValidateSet("none", "heatmap")]
  [string]$ProposalMode = "none",
  [ValidateSet("none", "iou")]
  [string]$QualityScoreMode = "none",
  [double]$Lr = 0.0001,
  [ValidateSet("constant", "cosine")]
  [string]$LrScheduler = "cosine",
  [int]$WarmupSteps = 500,
  [double]$MinLrRatio = 0.05,
  [double]$AugmentHFlipProb = 0.5,
  [double]$AugmentBrightness = 0.1,
  [double]$AugmentContrast = 0.1,
  [int]$NumWorkers = 4,
  [int]$PrefetchFactor = 4,
  [bool]$PersistentWorkers = $true,
  [int]$LogEvery = 10,
  [int]$SaveEverySteps = 50,
  [int]$KeepStepCheckpoints = 5,
  [bool]$Amp = $true,
  [bool]$ChannelsLast = $true,
  [bool]$CompileModel = $false,
  [bool]$Tf32 = $true,
  [bool]$CudnnBenchmark = $true,
  [bool]$SyncTiming = $false,
  [bool]$Ema = $true,
  [double]$EmaDecay = 0.999,
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Device = "auto",
  [double]$BoxWeight = 5.0,
  [double]$GiouWeight = 2.0,
  [double]$ObjWeight = 1.0,
  [double]$FutureWeight = 0.5,
  [double]$NoObjWeight = 0.1,
  [double]$ObjFocalGamma = 2.0,
  [double]$ObjFocalAlpha = 0.25,
  [double]$DensePositiveRadius = 0.0,
  [int]$DensePositiveTopk = 0,
  [int]$DenseHardNegativeTopk = 0,
  [double]$DenseRankWeight = 0.0,
  [double]$DenseRankMargin = 1.0,
  [int]$DenseRankNegativeTopk = 0,
  [ValidateSet("max", "all")]
  [string]$DenseRankPositiveMode = "max",
  [double]$ActionChunkConsistencyWeight = 0.0,
  [double]$MemoryQualityWeight = 0.0,
  [double]$MemoryQualitySigma = 0.08,
  [double]$MemoryQualityRecencyTau = 0.0,
  [bool]$MemoryQualityExcludeCurrent = $false,
  [double]$MotionObjWeight = 0.0,
  [double]$DenseHeatmapWeight = 0.0,
  [double]$DenseHeatmapSigma = 0.02,
  [double]$DenseHeatmapNegWeight = 0.02,
  [double]$DenseHeatmapFocalGamma = 2.0,
  [double]$MemoryMatchLossWeight = 0.0,
  [double]$QualityLossWeight = 0.0,
  [int]$QualityWarmupSteps = 0,
  [int]$QualityRampSteps = 0,
  [double]$QualityPositiveIou = 0.05,
  [int]$QualityHardNegativeTopk = 0,
  [double]$QualityFocalGamma = 1.0,
  [string]$DataRoot = "U:\URAP_datasets",
  [string]$FramesDir = "",
  [string]$OutDir = "",
  [string]$RunRoot = "",
  [int]$MinTrainFrames = 50000,
  [string]$ResumeWeights = "",
  [string]$InitWeights = "",
  [bool]$AutoResume = $true,
  [string]$CacheDir = "",
  [int]$Seed = -1
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
if ([string]::IsNullOrWhiteSpace($OutDir)) {
  $OutDir = Join-Path $Repo "artifacts\native_video_detector\$RunId"
}
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
  $RunRoot = Join-Path $Repo "artifacts\native_video_detector\$RunId"
}
$LogDir = Join-Path $RunRoot "logs"
$PidFile = Join-Path $RunRoot "train.pid"
$MetaFile = Join-Path $RunRoot "train_meta.json"
$StdoutLog = Join-Path $LogDir "train_stdout.log"
$StderrLog = Join-Path $LogDir "train_stderr.log"
if ([string]::IsNullOrWhiteSpace($FramesDir)) {
  $FramesDir = Join-Path $DataRoot "TransVisDrone\NPS\AllFrames\train"
}
$GtCsv = Join-Path $Repo "artifacts\nps_sota_research\tvd_nps_train_route_b_v3\gt.csv"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (-not (Test-Path $Python)) {
  throw "Python venv not found: $Python"
}
if (-not (Test-Path $FramesDir)) {
  throw "Frames dir not found: $FramesDir"
}
$FrameCount = (Get-ChildItem -LiteralPath $FramesDir -Filter "*.png" -File | Measure-Object).Count
if ($FrameCount -lt $MinTrainFrames) {
  throw "Frames dir appears incomplete: $FramesDir has $FrameCount png files, expected at least $MinTrainFrames. If a dataset migration is still running, wait for it to finish before training."
}
if (-not (Test-Path $GtCsv)) {
  throw "GT CSV not found: $GtCsv"
}

$ArgsList = @(
  "tools\train_native_video_detector.py",
  "--frames-dir", $FramesDir,
  "--gt-csv", $GtCsv,
  "--out-dir", $OutDir,
  "--epochs", "$Epochs",
  "--batch-size", "$BatchSize",
  "--image-size", "$ImageSize",
  "--clip-len", "$ClipLen",
  "--future-len", "$FutureLen",
  "--num-queries", "$NumQueries",
  "--d-model", "$DModel",
  "--nhead", "$Nhead",
  "--encoder-layers", "$EncoderLayers",
  "--decoder-layers", "$DecoderLayers",
  "--encoder-mode", "$EncoderMode",
  "--patch-stride", "$PatchStride",
  "--spatial-refine-layers", "$SpatialRefineLayers",
  "--spatial-refine-kernel", "$SpatialRefineKernel",
  "--spatial-refine-expansion", "$SpatialRefineExpansion",
  "--memory-mode", "$MemoryMode",
  "--box-size-scale", "$BoxSizeScale",
  "--query-mode", "$QueryMode",
  "--anchor-offset-cells", "$AnchorOffsetCells",
  "--dense-obj-source", "$DenseObjSource",
  "--memory-attention", "$MemoryAttention",
  "--memory-slots", "$MemorySlots",
  "--memory-match-mode", "$MemoryMatchMode",
  "--memory-match-weight", "$MemoryMatchWeight",
  "--memory-match-temperature", "$MemoryMatchTemperature",
  "--motion-score-mode", "$MotionScoreMode",
  "--motion-score-weight", "$MotionScoreWeight",
  "--proposal-mode", "$ProposalMode",
  "--quality-score-mode", "$QualityScoreMode",
  "--lr", "$Lr",
  "--lr-scheduler", "$LrScheduler",
  "--warmup-steps", "$WarmupSteps",
  "--min-lr-ratio", "$MinLrRatio",
  "--augment-hflip-prob", "$AugmentHFlipProb",
  "--augment-brightness", "$AugmentBrightness",
  "--augment-contrast", "$AugmentContrast",
  "--num-workers", "$NumWorkers",
  "--prefetch-factor", "$PrefetchFactor",
  "--box-weight", "$BoxWeight",
  "--giou-weight", "$GiouWeight",
  "--obj-weight", "$ObjWeight",
  "--future-weight", "$FutureWeight",
  "--noobj-weight", "$NoObjWeight",
  "--obj-focal-gamma", "$ObjFocalGamma",
  "--obj-focal-alpha", "$ObjFocalAlpha",
  "--dense-positive-radius", "$DensePositiveRadius",
  "--dense-positive-topk", "$DensePositiveTopk",
  "--dense-hard-negative-topk", "$DenseHardNegativeTopk",
  "--dense-rank-weight", "$DenseRankWeight",
  "--dense-rank-margin", "$DenseRankMargin",
  "--dense-rank-negative-topk", "$DenseRankNegativeTopk",
  "--dense-rank-positive-mode", "$DenseRankPositiveMode",
  "--action-chunk-consistency-weight", "$ActionChunkConsistencyWeight",
  "--memory-quality-weight", "$MemoryQualityWeight",
  "--memory-quality-sigma", "$MemoryQualitySigma",
  "--memory-quality-recency-tau", "$MemoryQualityRecencyTau",
  "--motion-obj-weight", "$MotionObjWeight",
  "--dense-heatmap-weight", "$DenseHeatmapWeight",
  "--dense-heatmap-sigma", "$DenseHeatmapSigma",
  "--dense-heatmap-neg-weight", "$DenseHeatmapNegWeight",
  "--dense-heatmap-focal-gamma", "$DenseHeatmapFocalGamma",
  "--memory-match-loss-weight", "$MemoryMatchLossWeight",
  "--quality-loss-weight", "$QualityLossWeight",
  "--quality-warmup-steps", "$QualityWarmupSteps",
  "--quality-ramp-steps", "$QualityRampSteps",
  "--quality-positive-iou", "$QualityPositiveIou",
  "--quality-hard-negative-topk", "$QualityHardNegativeTopk",
  "--quality-focal-gamma", "$QualityFocalGamma",
  "--device", "$Device",
  "--log-every", "$LogEvery",
  "--save-every-steps", "$SaveEverySteps",
  "--keep-step-checkpoints", "$KeepStepCheckpoints"
)
if ($MotionChannels) {
  $ArgsList += "--motion-channels"
}
if ($MemoryQualityExcludeCurrent) {
  $ArgsList += "--memory-quality-exclude-current"
}
if ($PersistentWorkers) {
  $ArgsList += "--persistent-workers"
}
if (-not [string]::IsNullOrWhiteSpace($CacheDir)) {
  if (-not (Test-Path $CacheDir)) {
    throw "Cache dir not found: $CacheDir"
  }
  $ArgsList += @("--cache-dir", $CacheDir)
}
if ($Seed -ge 0) {
  $ArgsList += @("--seed", "$Seed")
}
if ($Amp) {
  $ArgsList += "--amp"
}
if ($ChannelsLast) {
  $ArgsList += "--channels-last"
}
if ($CompileModel) {
  $ArgsList += "--compile"
}
if ($Tf32) {
  $ArgsList += "--tf32"
}
if ($CudnnBenchmark) {
  $ArgsList += "--cudnn-benchmark"
}
if ($SyncTiming) {
  $ArgsList += "--sync-timing"
}
if ($Ema) {
  $ArgsList += @("--ema", "--ema-decay", "$EmaDecay")
}
if (-not [string]::IsNullOrWhiteSpace($ResumeWeights) -and -not [string]::IsNullOrWhiteSpace($InitWeights)) {
  throw "ResumeWeights and InitWeights are mutually exclusive."
}
if (-not [string]::IsNullOrWhiteSpace($InitWeights)) {
  if (-not (Test-Path $InitWeights)) {
    throw "Init weights not found: $InitWeights"
  }
  $ArgsList += @("--init-weights", $InitWeights)
  $AutoResume = $false
}
if ([string]::IsNullOrWhiteSpace($ResumeWeights) -and $AutoResume) {
  $CandidateResume = Join-Path $OutDir "native_video_detector_latest.pt"
  if (Test-Path $CandidateResume) {
    $ResumeWeights = $CandidateResume
  }
}
if (-not [string]::IsNullOrWhiteSpace($ResumeWeights)) {
  if (-not (Test-Path $ResumeWeights)) {
    throw "Resume weights not found: $ResumeWeights"
  }
  $ArgsList += @("--resume", $ResumeWeights)
}

$Process = Start-Process -FilePath $Python `
  -ArgumentList $ArgsList `
  -WorkingDirectory $Repo `
  -RedirectStandardOutput $StdoutLog `
  -RedirectStandardError $StderrLog `
  -WindowStyle Hidden `
  -PassThru

Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ASCII
$Meta = [ordered]@{
  run_id = $RunId
  pid = $Process.Id
  started_at = (Get-Date).ToString("o")
  repo = $Repo
  python = $Python
  out_dir = $OutDir
  run_root = $RunRoot
  stdout_log = $StdoutLog
  stderr_log = $StderrLog
  frames_dir = $FramesDir
  frame_count = $FrameCount
  data_root = $DataRoot
  gt_csv = $GtCsv
  epochs = $Epochs
  batch_size = $BatchSize
  image_size = $ImageSize
  clip_len = $ClipLen
  future_len = $FutureLen
  num_queries = $NumQueries
  d_model = $DModel
  nhead = $Nhead
  encoder_layers = $EncoderLayers
  decoder_layers = $DecoderLayers
  encoder_mode = $EncoderMode
  patch_stride = $PatchStride
  spatial_refine_layers = $SpatialRefineLayers
  spatial_refine_kernel = $SpatialRefineKernel
  spatial_refine_expansion = $SpatialRefineExpansion
  motion_channels = $MotionChannels
  memory_mode = $MemoryMode
  box_size_scale = $BoxSizeScale
  query_mode = $QueryMode
  anchor_offset_cells = $AnchorOffsetCells
  dense_obj_source = $DenseObjSource
  memory_attention = $MemoryAttention
  memory_slots = $MemorySlots
  memory_match_mode = $MemoryMatchMode
  memory_match_weight = $MemoryMatchWeight
  memory_match_temperature = $MemoryMatchTemperature
  motion_score_mode = $MotionScoreMode
  motion_score_weight = $MotionScoreWeight
  proposal_mode = $ProposalMode
  quality_score_mode = $QualityScoreMode
  lr = $Lr
  lr_scheduler = $LrScheduler
  warmup_steps = $WarmupSteps
  min_lr_ratio = $MinLrRatio
  augment_hflip_prob = $AugmentHFlipProb
  augment_brightness = $AugmentBrightness
  augment_contrast = $AugmentContrast
  num_workers = $NumWorkers
  prefetch_factor = $PrefetchFactor
  persistent_workers = $PersistentWorkers
  log_every = $LogEvery
  save_every_steps = $SaveEverySteps
  amp = $Amp
  channels_last = $ChannelsLast
  compile_model = $CompileModel
  tf32 = $Tf32
  cudnn_benchmark = $CudnnBenchmark
  sync_timing = $SyncTiming
  ema = $Ema
  ema_decay = $EmaDecay
  device = $Device
  box_weight = $BoxWeight
  giou_weight = $GiouWeight
  obj_weight = $ObjWeight
  future_weight = $FutureWeight
  noobj_weight = $NoObjWeight
  obj_focal_gamma = $ObjFocalGamma
  obj_focal_alpha = $ObjFocalAlpha
  dense_positive_radius = $DensePositiveRadius
  dense_positive_topk = $DensePositiveTopk
  dense_hard_negative_topk = $DenseHardNegativeTopk
  dense_rank_weight = $DenseRankWeight
  dense_rank_margin = $DenseRankMargin
  dense_rank_negative_topk = $DenseRankNegativeTopk
  dense_rank_positive_mode = $DenseRankPositiveMode
  action_chunk_consistency_weight = $ActionChunkConsistencyWeight
  memory_quality_weight = $MemoryQualityWeight
  memory_quality_sigma = $MemoryQualitySigma
  memory_quality_recency_tau = $MemoryQualityRecencyTau
  memory_quality_exclude_current = $MemoryQualityExcludeCurrent
  motion_obj_weight = $MotionObjWeight
  dense_heatmap_weight = $DenseHeatmapWeight
  dense_heatmap_sigma = $DenseHeatmapSigma
  dense_heatmap_neg_weight = $DenseHeatmapNegWeight
  dense_heatmap_focal_gamma = $DenseHeatmapFocalGamma
  memory_match_loss_weight = $MemoryMatchLossWeight
  quality_loss_weight = $QualityLossWeight
  quality_warmup_steps = $QualityWarmupSteps
  quality_ramp_steps = $QualityRampSteps
  quality_positive_iou = $QualityPositiveIou
  quality_hard_negative_topk = $QualityHardNegativeTopk
  quality_focal_gamma = $QualityFocalGamma
  cache_dir = $CacheDir
  seed = if ($Seed -ge 0) { $Seed } else { $null }
  resume_weights = $ResumeWeights
  init_weights = $InitWeights
  auto_resume = $AutoResume
  command = "$Python $($ArgsList -join ' ')"
}
$Meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $MetaFile -Encoding UTF8

Write-Output "STARTED native video detector training"
Write-Output "PID: $($Process.Id)"
Write-Output "RunId: $RunId"
Write-Output "OutDir: $OutDir"
Write-Output "Stdout: $StdoutLog"
Write-Output "Stderr: $StderrLog"
Write-Output "Meta: $MetaFile"

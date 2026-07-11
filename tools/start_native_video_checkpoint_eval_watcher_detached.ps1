param(
  [string]$RunId = "nps_native_video_mvp_b32_w0_u_amp_ckpt",
  [string]$DataRoot = "U:\URAP_datasets",
  [string]$Split = "val",
  [int]$MaxSamples = 512,
  [int]$BatchSize = 16,
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
  [string]$PrimaryMetric = "map50",
  [string]$EvalName = "",
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Device = "auto",
  [int]$PollSeconds = 60,
  [int]$StableSeconds = 10,
  [int]$TimeoutMinutes = 0,
  [string]$WatcherRunId = "native_video_checkpoint_eval_watcher",
  [string]$WatcherSubdir = "checkpoint_eval_watcher",
  [string]$RunDir = "",
  [string]$RunRoot = "",
  [string]$WeightsName = "native_video_detector_latest.pt",
  [string]$WaitForFileName = "",
  [string]$CacheDir = "",
  [int]$CpuThreads = 4,
  [int]$NoEma = 0
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
if ([string]::IsNullOrWhiteSpace($RunDir)) {
  $RunDir = Join-Path $Repo "artifacts\native_video_detector\$RunId"
}
if ([string]::IsNullOrWhiteSpace($RunRoot)) {
  $RunRoot = $RunDir
}
$OutputRoot = Join-Path $RunRoot $WatcherSubdir
$LogDir = Join-Path $OutputRoot "logs"
$PidFile = Join-Path $OutputRoot "$WatcherRunId.pid"
$MetaFile = Join-Path $OutputRoot "$WatcherRunId.meta.json"
$RunnerFile = Join-Path $OutputRoot "$WatcherRunId.runner.ps1"
$Weights = Join-Path $RunDir $WeightsName
if ([string]::IsNullOrWhiteSpace($WaitForFileName)) {
  $WaitForFileName = $WeightsName
}
if ([string]::IsNullOrWhiteSpace($EvalName)) {
  $EvalName = "eval_$Split"
}
$WaitForFile = Join-Path $RunDir $WaitForFileName

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (Test-Path $PidFile) {
  $ExistingPid = Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($ExistingPid) {
    $Existing = Get-CimInstance Win32_Process -Filter "ProcessId = $ExistingPid" -ErrorAction SilentlyContinue
    if ($Existing -and $Existing.CommandLine -like "*$WatcherRunId.runner.ps1*") {
      Write-Output "Native video checkpoint eval watcher already running: pid=$ExistingPid"
      Get-Content -LiteralPath $MetaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$StartedAt = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $LogDir "watcher_${WatcherRunId}_${StartedAt}.out.log"
$Stderr = Join-Path $LogDir "watcher_${WatcherRunId}_${StartedAt}.err.log"

$Runner = @"
`$ErrorActionPreference = "Stop"
Set-Location "$Repo"
if ($CpuThreads -gt 0) {
  `$env:OMP_NUM_THREADS = "$CpuThreads"
  `$env:MKL_NUM_THREADS = "$CpuThreads"
  `$env:OPENBLAS_NUM_THREADS = "$CpuThreads"
  `$env:NUMEXPR_NUM_THREADS = "$CpuThreads"
  `$env:TORCH_NUM_THREADS = "$CpuThreads"
}
`$start = Get-Date
function Wait-StableFile {
  param(
    [Parameter(Mandatory = `$true)][string]`$Path,
    [int]`$StableSeconds = 10,
    [int]`$PollSeconds = 2
  )
  `$lastLength = -1L
  `$stableFor = 0
  while (`$true) {
    if ($TimeoutMinutes -gt 0 -and ((Get-Date) - `$start).TotalMinutes -gt $TimeoutMinutes) {
      throw "timeout waiting for stable file: `$Path"
    }
    if (-not (Test-Path `$Path)) {
      `$lastLength = -1L
      `$stableFor = 0
    } else {
      `$item = Get-Item -LiteralPath `$Path
      if (`$item.Length -gt 0 -and `$item.Length -eq `$lastLength) {
        `$stableFor += `$PollSeconds
        if (`$stableFor -ge `$StableSeconds) {
          return `$item
        }
      } else {
        `$lastLength = `$item.Length
        `$stableFor = 0
      }
    }
    Start-Sleep -Seconds `$PollSeconds
  }
}
Write-Output (@{
  kind = "native_video_eval_watcher_start"
  run_id = "$RunId"
  weights = "$Weights"
  wait_for_file = "$WaitForFile"
  split = "$Split"
} | ConvertTo-Json -Compress)
while (-not (Test-Path "$WaitForFile")) {
  if ($TimeoutMinutes -gt 0 -and ((Get-Date) - `$start).TotalMinutes -gt $TimeoutMinutes) {
    throw "timeout waiting for file: $WaitForFile"
  }
  Start-Sleep -Seconds $PollSeconds
}
if (-not (Test-Path "$Weights")) {
  throw "weights not found after wait file became ready: $Weights"
}
`$stableWeights = Wait-StableFile -Path "$Weights" -StableSeconds $StableSeconds -PollSeconds 2
Write-Output (@{
  kind = "native_video_eval_watcher_progress"
  stage = "weights_file_stable"
  wait_for_file = "$WaitForFile"
  weights = "$Weights"
  bytes = `$stableWeights.Length
} | ConvertTo-Json -Compress)
& "$Repo\tools\eval_native_video_checkpoint.ps1" -RunId "$RunId" -Weights "$Weights" -Split "$Split" -DataRoot "$DataRoot" -BatchSize $BatchSize -MaxSamples $MaxSamples -ScoreThreshold $ScoreThreshold -TopK $TopK -ProposalPrefilterTopK $ProposalPrefilterTopK -ProposalScoreWeight $ProposalScoreWeight -QualityScoreWeight $QualityScoreWeight -NmsIouThreshold $NmsIouThreshold -SamuraiMotionRerank $SamuraiMotionRerank -SamuraiAppearanceWeight $SamuraiAppearanceWeight -SamuraiMotionIouWeight $SamuraiMotionIouWeight -SamuraiCenterWeight $SamuraiCenterWeight -SamuraiCenterSigmaPixels $SamuraiCenterSigmaPixels -SamuraiUpdateScoreThreshold $SamuraiUpdateScoreThreshold -SamuraiUpdateMotionIouThreshold $SamuraiUpdateMotionIouThreshold -SamuraiLostTau $SamuraiLostTau -SamuraiVelocityMomentum $SamuraiVelocityMomentum -SamuraiTrackletRerank $SamuraiTrackletRerank -SamuraiTrackletCandidateTopK $SamuraiTrackletCandidateTopK -SamuraiTrackletMatchThreshold $SamuraiTrackletMatchThreshold -SamuraiTrackletMaxGap $SamuraiTrackletMaxGap -SamuraiTrackletSpawnScoreThreshold $SamuraiTrackletSpawnScoreThreshold -SamuraiTrackletLengthNorm $SamuraiTrackletLengthNorm -SamuraiTrackletAppearanceWeight $SamuraiTrackletAppearanceWeight -SamuraiTrackletWeight $SamuraiTrackletWeight -SamuraiTrackletUnmatchedScale $SamuraiTrackletUnmatchedScale -ActionChunkBackfill $ActionChunkBackfill -ActionChunkMaxStep $ActionChunkMaxStep -ActionChunkTopK $ActionChunkTopK -ActionChunkScoreDecay $ActionChunkScoreDecay -ActionChunkMergeMode "$ActionChunkMergeMode" -ActionChunkSupportIou $ActionChunkSupportIou -ActionChunkSupportWeight $ActionChunkSupportWeight -ActionChunkKeepUnmatched $ActionChunkKeepUnmatched -ExportLogEvery $ExportLogEvery -PrimaryMetric "$PrimaryMetric" -EvalName "$EvalName" -CacheDir "$CacheDir" -Device "$Device" -NoEma $NoEma
Write-Output (@{
  kind = "native_video_eval_watcher_done"
  run_id = "$RunId"
  split = "$Split"
} | ConvertTo-Json -Compress)
"@
$Runner | Set-Content -LiteralPath $RunnerFile -Encoding UTF8

$Proc = Start-Process -FilePath "powershell.exe" `
  -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $RunnerFile) `
  -WorkingDirectory $Repo `
  -RedirectStandardOutput $Stdout `
  -RedirectStandardError $Stderr `
  -PassThru `
  -WindowStyle Hidden

Set-Content -LiteralPath $PidFile -Value $Proc.Id -Encoding ASCII
$Meta = [ordered]@{
  started_at = (Get-Date).ToString("o")
  pid = $Proc.Id
  watcher_run_id = $WatcherRunId
  run_id = $RunId
  run_dir = $RunDir
  run_root = $RunRoot
  data_root = $DataRoot
  weights = $Weights
  wait_for_file = $WaitForFile
  split = $Split
  max_samples = $MaxSamples
  batch_size = $BatchSize
  score_threshold = $ScoreThreshold
  top_k = $TopK
  proposal_prefilter_top_k = $ProposalPrefilterTopK
  proposal_score_weight = $ProposalScoreWeight
  quality_score_weight = $QualityScoreWeight
  nms_iou_threshold = $NmsIouThreshold
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
  export_log_every = $ExportLogEvery
  primary_metric = $PrimaryMetric
  eval_name = $EvalName
  device = $Device
  cache_dir = $CacheDir
  cpu_threads = $CpuThreads
  no_ema = [bool]($NoEma -ne 0)
  poll_seconds = $PollSeconds
  stable_seconds = $StableSeconds
  timeout_minutes = $TimeoutMinutes
  watcher_subdir = $WatcherSubdir
  weights_name = $WeightsName
  wait_for_file_name = $WaitForFileName
  runner_file = $RunnerFile
  stdout_log = $Stdout
  stderr_log = $Stderr
}
$Meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $MetaFile -Encoding UTF8

Write-Output "Started native video checkpoint eval watcher."
Write-Output "PID: $($Proc.Id)"
Write-Output "RunId: $RunId"
Write-Output "Weights: $Weights"
Write-Output "WaitForFile: $WaitForFile"
Write-Output "Stdout: $Stdout"
Write-Output "Stderr: $Stderr"
Write-Output "Meta: $MetaFile"

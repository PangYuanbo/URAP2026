param(
  [string]$RunId = "nps_native_video_mvp_b32_w0_u_amp_ckpt",
  [string]$DataRoot = "U:\URAP_datasets",
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
  [int]$MinTrainFrames = 50000,
  [int]$EvalMaxSamples = 512,
  [string]$EvalSplit = "val",
  [string]$CacheDir = "",
  [string]$EvalCacheDir = "",
  [string]$FinalTestCacheDir = "",
  [bool]$UseFrameCache = $true,
  [bool]$StartFrameCacheIfMissing = $true,
  [bool]$ChannelsLast = $true,
  [bool]$CompileModel = $false,
  [bool]$Tf32 = $true,
  [bool]$CudnnBenchmark = $true,
  [bool]$SyncTiming = $false,
  [bool]$Ema = $true,
  [double]$EmaDecay = 0.999,
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$Device = "auto",
  [ValidateSet("auto", "cpu", "cuda")]
  [string]$EvalDevice = "auto",
  [double]$BoxWeight = 5.0,
  [double]$GiouWeight = 2.0,
  [double]$ObjWeight = 1.0,
  [double]$FutureWeight = 0.5,
  [double]$NoObjWeight = 0.1,
  [double]$ObjFocalGamma = 2.0,
  [double]$ObjFocalAlpha = 0.25,
  [bool]$StartContinuousValWatcher = $true,
  [bool]$StartBestValTestWatcher = $true,
  [bool]$StartFinalTestWatcher = $false,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Get-SplitFramesDir {
  param([string]$Split)
  return (Join-Path $DataRoot "TransVisDrone\NPS\AllFrames\$Split")
}

function Test-FrameCacheReady {
  param(
    [string]$Split,
    [string]$FramesDir,
    [string]$SplitCacheDir
  )
  $FrameCount = 0
  if (Test-Path $FramesDir) {
    $FrameCount = (Get-ChildItem -LiteralPath $FramesDir -Filter "Clip_*_*.png" -File | Measure-Object).Count
  }
  $SummaryPath = Join-Path $SplitCacheDir "cache_summary.json"
  $CachedFrameCount = 0
  if (Test-Path $SplitCacheDir) {
    $CachedFrameCount = (Get-ChildItem -LiteralPath $SplitCacheDir -Filter "Clip_*_*.pt" -File | Measure-Object).Count
  }
  $Ready = $false
  if (Test-Path $SummaryPath) {
    try {
      $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
      $Ready = ([int]$Summary.total -ge $FrameCount) -and ([int]$Summary.image_size -eq $ImageSize) -and ($CachedFrameCount -ge $FrameCount)
    } catch {
      $Ready = $false
    }
  }
  return [ordered]@{
    split = $Split
    frames_dir = $FramesDir
    cache_dir = $SplitCacheDir
    frame_count = $FrameCount
    cached_frame_count = $CachedFrameCount
    summary = $SummaryPath
    ready = $Ready
  }
}

function Test-FrameCacheBuilderRunning {
  param([string]$CacheRunId)
  $OutDir = Join-Path $Repo "artifacts\native_video_detector\$CacheRunId"
  $PidFile = Join-Path $OutDir "frame_cache.pid"
  if (-not (Test-Path $PidFile)) {
    return $false
  }
  try {
    $PidValue = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
    $ProcInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $PidValue" -ErrorAction SilentlyContinue
    return ($null -ne $ProcInfo) -and ([string]$ProcInfo.CommandLine -like "*build_native_video_frame_cache.py*")
  } catch {
    return $false
  }
}

if ([string]::IsNullOrWhiteSpace($CacheDir)) {
  $CacheDir = Join-Path $DataRoot "TransVisDrone\NPS\AllFrames\train_cache_${ImageSize}"
}
if ([string]::IsNullOrWhiteSpace($EvalCacheDir)) {
  $EvalCacheDir = Join-Path $DataRoot "TransVisDrone\NPS\AllFrames\val_cache_${ImageSize}"
}
if ([string]::IsNullOrWhiteSpace($FinalTestCacheDir)) {
  $FinalTestCacheDir = Join-Path $DataRoot "TransVisDrone\NPS\AllFrames\test_cache_${ImageSize}"
}

if ($DryRun) {
  Write-Output "DRY RUN: native video post-migration pipeline"
  Write-Output "No dataset readiness check, cache builder, training, or watcher process was started."
  Write-Output "RunId: $RunId"
  Write-Output "DataRoot: $DataRoot"
  Write-Output "TrainFramesDir: $(Get-SplitFramesDir -Split 'train')"
  Write-Output "ValFramesDir: $(Get-SplitFramesDir -Split 'val')"
  Write-Output "TestFramesDir: $(Get-SplitFramesDir -Split 'test')"
  Write-Output "TrainCacheDir: $CacheDir"
  Write-Output "ValCacheDir: $EvalCacheDir"
  Write-Output "TestCacheDir: $FinalTestCacheDir"
  Write-Output "Architecture: clip_len=$ClipLen future_len=$FutureLen num_queries=$NumQueries d_model=$DModel nhead=$Nhead encoder_layers=$EncoderLayers decoder_layers=$DecoderLayers encoder_mode=$EncoderMode patch_stride=$PatchStride"
  Write-Output "Training: epochs=$Epochs batch_size=$BatchSize image_size=$ImageSize device=$Device amp=True channels_last=$ChannelsLast tf32=$Tf32 cudnn_benchmark=$CudnnBenchmark"
  Write-Output "Evaluation: eval_split=$EvalSplit eval_max_samples=$EvalMaxSamples eval_device=$EvalDevice best_val_test_full_split=$StartBestValTestWatcher final_test_watcher=$StartFinalTestWatcher"
  Write-Output "MVP audit JSON:"
  Write-Output (Join-Path $Repo "artifacts\native_video_detector\$RunId\mvp_audit.json")
  Write-Output "Monitor commands:"
  Write-Output "powershell -ExecutionPolicy Bypass -File tools\monitor_native_video_detector_train.ps1 -RunId $RunId -Tail 20"
  Write-Output "powershell -ExecutionPolicy Bypass -File tools\monitor_native_video_best_val_test_watcher.ps1 -RunId $RunId -Tail 20"
  return
}

Write-Output "Checking native video dataset readiness..."
& (Join-Path $Repo "tools\check_native_video_dataset_ready.ps1") `
  -DataRoot $DataRoot `
  -MinTrainFrames $MinTrainFrames
if ($LASTEXITCODE -ne 0) {
  throw "Dataset readiness check failed. Do not start training while migration is incomplete."
}

if ($UseFrameCache) {
  $TrainFramesDir = Get-SplitFramesDir -Split "train"
  $ValFramesDir = Get-SplitFramesDir -Split "val"
  $TestFramesDir = Get-SplitFramesDir -Split "test"
  $CacheChecks = @(
    (Test-FrameCacheReady -Split "train" -FramesDir $TrainFramesDir -SplitCacheDir $CacheDir),
    (Test-FrameCacheReady -Split "val" -FramesDir $ValFramesDir -SplitCacheDir $EvalCacheDir),
    (Test-FrameCacheReady -Split "test" -FramesDir $TestFramesDir -SplitCacheDir $FinalTestCacheDir)
  )
  $MissingCaches = @($CacheChecks | Where-Object { -not $_.ready })
  foreach ($Check in $CacheChecks) {
    Write-Output "cache_check split=$($Check.split) ready=$($Check.ready) frames=$($Check.frame_count) cached=$($Check.cached_frame_count) cache=$($Check.cache_dir) summary=$($Check.summary)"
  }
  if ($MissingCaches.Count -gt 0) {
    if (-not $StartFrameCacheIfMissing) {
      throw "Frame cache is incomplete and StartFrameCacheIfMissing is false. Do not start training until cache is ready."
    }
    foreach ($Check in $MissingCaches) {
      $CacheRunId = "nps_native_video_frame_cache_$($Check.split)_${ImageSize}"
      if (Test-FrameCacheBuilderRunning -CacheRunId $CacheRunId) {
        Write-Output "Frame cache builder already running for split=$($Check.split) RunId=$CacheRunId"
      } else {
        Write-Output "Starting frame cache builder for split=$($Check.split)..."
        & (Join-Path $Repo "tools\start_native_video_frame_cache_detached.ps1") `
          -RunId $CacheRunId `
          -DataRoot $DataRoot `
          -FramesDir $Check.frames_dir `
          -CacheDir $Check.cache_dir `
          -ImageSize $ImageSize `
          -MaxFrames 0 `
          -LogEvery 1000
      }
    }
    Write-Output "Frame cache preparation is not complete; training was not started."
    Write-Output "Re-run this pipeline after cache monitors report READY summaries."
    Write-Output "Train cache monitor: powershell -ExecutionPolicy Bypass -File tools\monitor_native_video_frame_cache.ps1 -RunId nps_native_video_frame_cache_train_${ImageSize} -Tail 20"
    Write-Output "Val cache monitor: powershell -ExecutionPolicy Bypass -File tools\monitor_native_video_frame_cache.ps1 -RunId nps_native_video_frame_cache_val_${ImageSize} -Tail 20"
    Write-Output "Test cache monitor: powershell -ExecutionPolicy Bypass -File tools\monitor_native_video_frame_cache.ps1 -RunId nps_native_video_frame_cache_test_${ImageSize} -Tail 20"
    return
  }
}

Write-Output "Starting native video detector training..."
& (Join-Path $Repo "tools\start_native_video_detector_train_detached.ps1") `
  -RunId $RunId `
  -DataRoot $DataRoot `
  -Epochs $Epochs `
  -BatchSize $BatchSize `
  -ImageSize $ImageSize `
  -ClipLen $ClipLen `
  -FutureLen $FutureLen `
  -NumQueries $NumQueries `
  -DModel $DModel `
  -Nhead $Nhead `
  -EncoderLayers $EncoderLayers `
  -DecoderLayers $DecoderLayers `
  -EncoderMode $EncoderMode `
  -PatchStride $PatchStride `
  -Lr $Lr `
  -LrScheduler $LrScheduler `
  -WarmupSteps $WarmupSteps `
  -MinLrRatio $MinLrRatio `
  -AugmentHFlipProb $AugmentHFlipProb `
  -AugmentBrightness $AugmentBrightness `
  -AugmentContrast $AugmentContrast `
  -NumWorkers $NumWorkers `
  -PrefetchFactor $PrefetchFactor `
  -PersistentWorkers $PersistentWorkers `
  -LogEvery $LogEvery `
  -SaveEverySteps $SaveEverySteps `
  -MinTrainFrames $MinTrainFrames `
  -CacheDir $CacheDir `
  -ChannelsLast $ChannelsLast `
  -CompileModel $CompileModel `
  -Tf32 $Tf32 `
  -CudnnBenchmark $CudnnBenchmark `
  -SyncTiming $SyncTiming `
  -Ema $Ema `
  -EmaDecay $EmaDecay `
  -Device $Device `
  -BoxWeight $BoxWeight `
  -GiouWeight $GiouWeight `
  -ObjWeight $ObjWeight `
  -FutureWeight $FutureWeight `
  -NoObjWeight $NoObjWeight `
  -ObjFocalGamma $ObjFocalGamma `
  -ObjFocalAlpha $ObjFocalAlpha

Write-Output "Starting native video checkpoint eval watcher..."
& (Join-Path $Repo "tools\start_native_video_checkpoint_eval_watcher_detached.ps1") `
  -RunId $RunId `
  -DataRoot $DataRoot `
  -Split $EvalSplit `
  -MaxSamples $EvalMaxSamples `
  -WatcherRunId "native_video_early_eval_watcher" `
  -WatcherSubdir "early_eval_watcher" `
  -WeightsName "native_video_detector_latest.pt" `
  -WaitForFileName "native_video_detector_latest.pt" `
  -CacheDir $EvalCacheDir `
  -Device $EvalDevice

if ($StartContinuousValWatcher) {
  Write-Output "Starting native video continuous val watcher..."
  & (Join-Path $Repo "tools\start_native_video_continuous_val_watcher_detached.ps1") `
    -RunId $RunId `
    -DataRoot $DataRoot `
    -MaxSamples $EvalMaxSamples `
    -BatchSize 16 `
    -ScoreThreshold 0.0 `
    -TopK 32 `
    -NmsIouThreshold 0.5 `
    -PrimaryMetric "map50" `
    -PollSeconds 120 `
    -CacheDir $EvalCacheDir `
    -Device $EvalDevice
}

if ($StartFinalTestWatcher) {
  Write-Output "Starting native video final-model test eval watcher..."
  & (Join-Path $Repo "tools\start_native_video_checkpoint_eval_watcher_detached.ps1") `
    -RunId $RunId `
    -DataRoot $DataRoot `
    -Split "test" `
    -MaxSamples 0 `
    -WatcherRunId "native_video_final_test_eval_watcher" `
    -WatcherSubdir "final_test_eval_watcher" `
    -WeightsName "native_video_detector.pt" `
    -WaitForFileName "summary.json" `
    -CacheDir $FinalTestCacheDir `
    -Device $EvalDevice
}

if ($StartBestValTestWatcher) {
  Write-Output "Starting native video best-val test eval watcher..."
  & (Join-Path $Repo "tools\start_native_video_best_val_test_watcher_detached.ps1") `
    -RunId $RunId `
    -DataRoot $DataRoot `
    -MaxSamples 0 `
    -BatchSize 16 `
    -ScoreThreshold 0.0 `
    -TopK 32 `
    -NmsIouThreshold 0.5 `
    -PrimaryMetric "map50" `
    -PollSeconds 120 `
    -CacheDir $FinalTestCacheDir `
    -Device $EvalDevice
}

Write-Output "Native video post-migration pipeline launch complete."
Write-Output "Train monitor:"
Write-Output "powershell -ExecutionPolicy Bypass -File tools\monitor_native_video_detector_train.ps1 -RunId $RunId -Tail 20"
Write-Output "Eval watcher monitor:"
Write-Output "powershell -ExecutionPolicy Bypass -File tools\monitor_native_video_checkpoint_eval_watcher.ps1 -RunId $RunId -WatcherRunId native_video_early_eval_watcher -WatcherSubdir early_eval_watcher -Tail 20"
Write-Output "Continuous val watcher monitor:"
Write-Output "powershell -ExecutionPolicy Bypass -File tools\monitor_native_video_continuous_val_watcher.ps1 -RunId $RunId -Tail 20"
Write-Output "Best-val test watcher monitor:"
Write-Output "powershell -ExecutionPolicy Bypass -File tools\monitor_native_video_best_val_test_watcher.ps1 -RunId $RunId -Tail 20"
Write-Output "MVP audit JSON:"
Write-Output (Join-Path $Repo "artifacts\native_video_detector\$RunId\mvp_audit.json")
Write-Output "Final-model test watcher monitor:"
Write-Output "powershell -ExecutionPolicy Bypass -File tools\monitor_native_video_checkpoint_eval_watcher.ps1 -RunId $RunId -WatcherRunId native_video_final_test_eval_watcher -WatcherSubdir final_test_eval_watcher -Tail 20"

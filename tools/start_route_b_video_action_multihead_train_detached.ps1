param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$TrackletJsonl = 'artifacts\route_b_official\aot_part0_tvd_val\tvd_aot_part0_conf0p2_ioutrack\route_b_tracklets_min2_gap2.jsonl',
  [string]$FrameRoot = 'D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest\test\part0\frames',
  [string]$Out = 'artifacts\route_b_official\aot_part0_video_action_multihead_train\video_action_multihead.pt',
  [int]$PastLen = 2,
  [int]$FutureLen = 1,
  [int]$CropSize = 48,
  [double]$CropScale = 4.0,
  [int]$ImageWidth = 2432,
  [int]$ImageHeight = 2048,
  [int]$MinTrackletRows = 3,
  [int]$MaxSamples = 0,
  [int]$Epochs = 5,
  [int]$BatchSize = 16,
  [int]$DModel = 64,
  [int]$NHead = 4,
  [int]$NumLayers = 2,
  [double]$Lr = 0.001,
  [string]$ConfidenceTarget = 'max',
  [double]$ConfidenceLossWeight = 1.0,
  [int]$NumWorkers = 0,
  [int]$FrameCacheSize = 8,
  [string]$RunId = 'route_b_video_action_multihead_train',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\route_b_official\aot_part0_video_action_multihead_train_runner')
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -Path $TrackletJsonl -PathType Leaf)) { throw "TrackletJsonl not found: $TrackletJsonl" }
if (-not (Test-Path -Path $FrameRoot -PathType Container)) { throw "FrameRoot not found: $FrameRoot" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*train-video-action-multihead-policy*') {
      Write-Host "Route B video-action multihead train already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$outDir = Split-Path -Parent $Out
if ($outDir) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logDir "runner_${RunId}_${ts}.err.txt"

$argList = @(
  '-m', 'qstr_dronedet.cli', 'train-video-action-multihead-policy',
  '--tracklet-jsonl', $TrackletJsonl,
  '--out', $Out,
  '--frame-root', $FrameRoot,
  '--past-len', [string]$PastLen,
  '--future-len', [string]$FutureLen,
  '--crop-size', [string]$CropSize,
  '--crop-scale', [string]$CropScale,
  '--image-width', [string]$ImageWidth,
  '--image-height', [string]$ImageHeight,
  '--min-tracklet-rows', [string]$MinTrackletRows,
  '--epochs', [string]$Epochs,
  '--batch-size', [string]$BatchSize,
  '--d-model', [string]$DModel,
  '--nhead', [string]$NHead,
  '--num-layers', [string]$NumLayers,
  '--lr', [string]$Lr,
  '--confidence-target', $ConfidenceTarget,
  '--confidence-loss-weight', [string]$ConfidenceLossWeight,
  '--num-workers', [string]$NumWorkers,
  '--frame-cache-size', [string]$FrameCacheSize
)
if ($MaxSamples -gt 0) {
  $argList += @('--max-samples', [string]$MaxSamples)
}

$env:PYTHONPATH = $RepoRoot
$proc = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "python=$Python",
  "run_id=$RunId",
  "repo_root=$RepoRoot",
  "tracklet_jsonl=$TrackletJsonl",
  "frame_root=$FrameRoot",
  "out=$Out",
  "past_len=$PastLen",
  "future_len=$FutureLen",
  "crop_size=$CropSize",
  "crop_scale=$CropScale",
  "image_width=$ImageWidth",
  "image_height=$ImageHeight",
  "min_tracklet_rows=$MinTrackletRows",
  "max_samples=$MaxSamples",
  "epochs=$Epochs",
  "batch_size=$BatchSize",
  "d_model=$DModel",
  "nhead=$NHead",
  "num_layers=$NumLayers",
  "lr=$Lr",
  "confidence_target=$ConfidenceTarget",
  "confidence_loss_weight=$ConfidenceLossWeight",
  "num_workers=$NumWorkers",
  "frame_cache_size=$FrameCacheSize",
  "output_root=$OutputRoot",
  "stdout=$stdout",
  "stderr=$stderr",
  "cmd_args=$($argList -join ' ')"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host 'Started detached Route B video-action multihead training.'
Get-Content $metaFile

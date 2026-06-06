param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$TrackletJsonl,
  [string]$FrameRoot = '',
  [string]$Out,
  [int]$PastLen = 7,
  [int]$FutureLen = 2,
  [string[]]$Horizons = @('3', '5', '7'),
  [int]$CropSize = 64,
  [double]$CropScale = 4.0,
  [int]$ImageWidth = 1920,
  [int]$ImageHeight = 1280,
  [int]$MinTrackletRows = 9,
  [int]$MaxSamples = 0,
  [int]$Epochs = 8,
  [int]$BatchSize = 256,
  [int]$DModel = 128,
  [int]$NHead = 4,
  [int]$NumLayers = 4,
  [double]$Lr = 0.0005,
  [double]$ActionLossWeight = 0.15,
  [string]$MotionPosWeight = 'auto',
  [int]$NumWorkers = 2,
  [int]$FrameCacheSize = 64,
  [int]$MinCFreeGb = 20,
  [switch]$NoPinMemory,
  [switch]$NoShuffle,
  [switch]$DisableCrops,
  [switch]$AllowMissingImages,
  [string]$RunId = 'ego_adaptive_vatd_train',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\ego_adaptive_vatd\train_runner')
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not $TrackletJsonl) { throw 'TrackletJsonl is required.' }
if (-not $Out) { throw 'Out is required.' }
if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -Path $TrackletJsonl -PathType Leaf)) { throw "TrackletJsonl not found: $TrackletJsonl" }
if ($FrameRoot -and -not (Test-Path -Path $FrameRoot -PathType Container)) { throw "FrameRoot not found: $FrameRoot" }
if ($BatchSize -lt 1) { throw 'BatchSize must be >= 1.' }
if ($NumWorkers -lt 0) { throw 'NumWorkers must be >= 0.' }
$parsedHorizons = @()
foreach ($item in $Horizons) {
  foreach ($part in ([string]$item -split ',')) {
    if ($part.Trim()) { $parsedHorizons += [int]$part.Trim() }
  }
}
if (-not $parsedHorizons -or $parsedHorizons.Count -lt 1) { throw 'At least one horizon is required.' }
foreach ($h in $parsedHorizons) {
  if ($h -lt 1 -or $h -gt $PastLen) { throw "Invalid horizon $h; each horizon must be in [1, PastLen=$PastLen]." }
}

$cDrive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'" -ErrorAction SilentlyContinue
if ($cDrive) {
  $cFreeGb = [math]::Round(($cDrive.FreeSpace / 1GB), 2)
  if ($cFreeGb -lt $MinCFreeGb) {
    throw "C: free space ${cFreeGb}GB is below MinCFreeGb=${MinCFreeGb}GB. Move outputs off C: or lower MinCFreeGb deliberately."
  }
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*train-ego-adaptive-vatd-policy*') {
      Write-Host "Ego-Adaptive VATD training already running: pid=$existingPid"
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
  '-m', 'qstr_dronedet.cli', 'train-ego-adaptive-vatd-policy',
  '--tracklet-jsonl', $TrackletJsonl,
  '--out', $Out,
  '--past-len', [string]$PastLen,
  '--future-len', [string]$FutureLen,
  '--horizons'
)
foreach ($h in $parsedHorizons) { $argList += [string]$h }
$argList += @(
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
  '--action-loss-weight', [string]$ActionLossWeight,
  '--motion-pos-weight', $MotionPosWeight,
  '--num-workers', [string]$NumWorkers,
  '--frame-cache-size', [string]$FrameCacheSize
)
if ($FrameRoot) { $argList += @('--frame-root', $FrameRoot) }
if ($MaxSamples -gt 0) { $argList += @('--max-samples', [string]$MaxSamples) }
if ($NoPinMemory) { $argList += '--no-pin-memory' }
if ($NoShuffle) { $argList += '--no-shuffle' }
if ($DisableCrops) { $argList += '--disable-crops' }
if ($AllowMissingImages) { $argList += '--allow-missing-images' }

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
  "horizons=$($parsedHorizons -join ',')",
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
  "action_loss_weight=$ActionLossWeight",
  "motion_pos_weight=$MotionPosWeight",
  "num_workers=$NumWorkers",
  "frame_cache_size=$FrameCacheSize",
  "no_pin_memory=$NoPinMemory",
  "no_shuffle=$NoShuffle",
  "disable_crops=$DisableCrops",
  "allow_missing_images=$AllowMissingImages",
  "output_root=$OutputRoot",
  "stdout=$stdout",
  "stderr=$stderr",
  "cmd_args=$($argList -join ' ')",
  "c_free_gb=$([math]::Round(((Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace / 1GB), 2))",
  "min_c_free_gb=$MinCFreeGb"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host 'Started detached Ego-Adaptive VATD training.'
Get-Content $metaFile

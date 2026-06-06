param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$TrackletJsonl,
  [string]$Weights,
  [string]$Out,
  [string]$FrameRoot = '',
  [string]$ImageNameTemplate = '{seq}_{frame_id_05d}.png',
  [double]$ErrorScale = 0.2,
  [int]$MinTrackletRows = 9,
  [int]$MaxSamples = 0,
  [int]$BatchSize = 256,
  [int]$NumWorkers = 4,
  [int]$FrameCacheSize = 64,
  [string]$FusionMode = 'motion_action',
  [switch]$AllowMissingImages,
  [switch]$DisableCrops,
  [string]$RunId = 'ego_adaptive_vatd_score',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\ego_adaptive_vatd\score_runner')
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -Path $TrackletJsonl -PathType Leaf)) { throw "TrackletJsonl not found: $TrackletJsonl" }
if (-not (Test-Path -Path $Weights -PathType Leaf)) { throw "Weights not found: $Weights" }
if ($FrameRoot -and -not (Test-Path -Path $FrameRoot -PathType Container)) { throw "FrameRoot not found: $FrameRoot" }
if (-not $Out) { throw 'Out must be provided' }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*score-ego-adaptive-vatd-tracklets*') {
      Write-Host "Ego-Adaptive VATD score already running: pid=$existingPid"
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
  '-m', 'qstr_dronedet.cli', 'score-ego-adaptive-vatd-tracklets',
  '--tracklet-jsonl', $TrackletJsonl,
  '--weights', $Weights,
  '--out', $Out,
  '--error-scale', [string]$ErrorScale,
  '--min-tracklet-rows', [string]$MinTrackletRows,
  '--batch-size', [string]$BatchSize,
  '--num-workers', [string]$NumWorkers,
  '--frame-cache-size', [string]$FrameCacheSize,
  '--fusion-mode', $FusionMode
)
if ($FrameRoot) { $argList += @('--frame-root', $FrameRoot) }
if ($ImageNameTemplate) { $argList += @('--image-name-template', $ImageNameTemplate) }
if ($MaxSamples -gt 0) { $argList += @('--max-samples', [string]$MaxSamples) }
if ($AllowMissingImages) { $argList += '--allow-missing-images' }
if ($DisableCrops) { $argList += '--disable-crops' }

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
  "weights=$Weights",
  "out=$Out",
  "frame_root=$FrameRoot",
  "image_name_template=$ImageNameTemplate",
  "error_scale=$ErrorScale",
  "min_tracklet_rows=$MinTrackletRows",
  "max_samples=$MaxSamples",
  "batch_size=$BatchSize",
  "num_workers=$NumWorkers",
  "frame_cache_size=$FrameCacheSize",
  "fusion_mode=$FusionMode",
  "allow_missing_images=$AllowMissingImages",
  "disable_crops=$DisableCrops",
  "output_root=$OutputRoot",
  "stdout=$stdout",
  "stderr=$stderr",
  "cmd_args=$($argList -join ' ')"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host 'Started detached Ego-Adaptive VATD scoring.'
Get-Content $metaFile

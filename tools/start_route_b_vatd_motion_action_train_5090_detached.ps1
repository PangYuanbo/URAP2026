param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Python = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$TrackletJsonl,
  [string]$FrameRoot = '',
  [string]$Out,
  [int]$PastLen = 3,
  [int]$FutureLen = 1,
  [int]$CropSize = 64,
  [double]$CropScale = 4.0,
  [int]$ImageWidth = 1920,
  [int]$ImageHeight = 1080,
  [int]$MinTrackletRows = 4,
  [int]$MaxSamples = 0,
  [int]$Epochs = 6,
  [int]$BatchSize = 4096,
  [int]$DModel = 384,
  [int]$NHead = 8,
  [int]$NumLayers = 8,
  [double]$Lr = 0.001,
  [double]$ActionLossWeight = 0.25,
  [string]$MotionPosWeight = 'auto',
  [int]$NumWorkers = 8,
  [int]$FrameCacheSize = 512,
  [int]$MinCFreeGb = 20,
  [switch]$NoShuffle,
  [switch]$DisableCrops,
  [switch]$AllowMissingImages,
  [string]$RunId = 'route_b_vatd_motion_action_train_5090',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\route_b_official\vatd_motion_action_train_5090_runner')
)

$ErrorActionPreference = 'Stop'
Set-Location $RepoRoot

if (-not $TrackletJsonl) { throw 'TrackletJsonl is required.' }
if (-not $Out) { throw 'Out is required.' }
if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -Path $TrackletJsonl -PathType Leaf)) { throw "TrackletJsonl not found: $TrackletJsonl" }
if ($FrameRoot -and -not (Test-Path -Path $FrameRoot -PathType Container)) { throw "FrameRoot not found: $FrameRoot" }
if ($BatchSize -lt 1) { throw 'BatchSize must be >= 1.' }
if ($NumWorkers -lt 1) { throw '5090 preset expects NumWorkers >= 1. Use start_route_b_vatd_motion_action_train_detached.ps1 for CPU-only/serial IO runs.' }

$cDrive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'" -ErrorAction SilentlyContinue
if ($cDrive) {
  $cFreeGb = [math]::Round(($cDrive.FreeSpace / 1GB), 2)
  if ($cFreeGb -lt $MinCFreeGb) {
    throw "C: free space ${cFreeGb}GB is below MinCFreeGb=${MinCFreeGb}GB. Move outputs off C: or lower MinCFreeGb deliberately."
  }
}

$args = @(
  '-RepoRoot', $RepoRoot,
  '-Python', $Python,
  '-TrackletJsonl', $TrackletJsonl,
  '-Out', $Out,
  '-PastLen', [string]$PastLen,
  '-FutureLen', [string]$FutureLen,
  '-CropSize', [string]$CropSize,
  '-CropScale', [string]$CropScale,
  '-ImageWidth', [string]$ImageWidth,
  '-ImageHeight', [string]$ImageHeight,
  '-MinTrackletRows', [string]$MinTrackletRows,
  '-Epochs', [string]$Epochs,
  '-BatchSize', [string]$BatchSize,
  '-DModel', [string]$DModel,
  '-NHead', [string]$NHead,
  '-NumLayers', [string]$NumLayers,
  '-Lr', [string]$Lr,
  '-ActionLossWeight', [string]$ActionLossWeight,
  '-MotionPosWeight', $MotionPosWeight,
  '-NumWorkers', [string]$NumWorkers,
  '-FrameCacheSize', [string]$FrameCacheSize,
  '-RunId', $RunId,
  '-OutputRoot', $OutputRoot
)
if ($FrameRoot) { $args += @('-FrameRoot', $FrameRoot) }
if ($MaxSamples -gt 0) { $args += @('-MaxSamples', [string]$MaxSamples) }
if ($NoShuffle) { $args += '-NoShuffle' }
if ($DisableCrops) { $args += '-DisableCrops' }
if ($AllowMissingImages) { $args += '-AllowMissingImages' }

& (Join-Path $PSScriptRoot 'start_route_b_vatd_motion_action_train_detached.ps1') @args

$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"
if (Test-Path $metaFile) {
  @(
    "preset=5090",
    "preset_note=high-throughput VATD/OCTO-style motion-action training; pin memory enabled by omission of -NoPinMemory",
    "c_free_gb=$([math]::Round(((Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'").FreeSpace / 1GB), 2))",
    "min_c_free_gb=$MinCFreeGb"
  ) | Add-Content -Path $metaFile -Encoding utf8
}

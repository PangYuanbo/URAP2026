param(
  [string]$RunId = "nps_native_video_frame_cache",
  [string]$DataRoot = "U:\URAP_datasets",
  [string]$FramesDir = "",
  [string]$CacheDir = "",
  [int]$ImageSize = 320,
  [int]$MaxFrames = 0,
  [int]$LogEvery = 1000,
  [bool]$Overwrite = $false
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Python = Join-Path $Repo "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$OutDir = Join-Path $Repo "artifacts\native_video_detector\$RunId"
$LogDir = Join-Path $OutDir "logs"
$PidFile = Join-Path $OutDir "frame_cache.pid"
$MetaFile = Join-Path $OutDir "frame_cache_meta.json"
$StdoutLog = Join-Path $LogDir "frame_cache_stdout.log"
$StderrLog = Join-Path $LogDir "frame_cache_stderr.log"

if ([string]::IsNullOrWhiteSpace($FramesDir)) {
  $FramesDir = Join-Path $DataRoot "TransVisDrone\NPS\AllFrames\train"
}
if ([string]::IsNullOrWhiteSpace($CacheDir)) {
  $CacheDir = Join-Path $DataRoot "TransVisDrone\NPS\AllFrames\train_cache_${ImageSize}"
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path $Python)) {
  throw "Python venv not found: $Python"
}
if (-not (Test-Path $FramesDir)) {
  throw "Frames dir not found: $FramesDir"
}

$FrameCount = (Get-ChildItem -LiteralPath $FramesDir -Filter "Clip_*_*.png" -File | Measure-Object).Count
$ArgsList = @(
  "tools\build_native_video_frame_cache.py",
  "--frames-dir", $FramesDir,
  "--cache-dir", $CacheDir,
  "--image-size", "$ImageSize",
  "--max-frames", "$MaxFrames",
  "--log-every", "$LogEvery"
)
if ($Overwrite) {
  $ArgsList += "--overwrite"
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
  stdout_log = $StdoutLog
  stderr_log = $StderrLog
  frames_dir = $FramesDir
  cache_dir = $CacheDir
  image_size = $ImageSize
  max_frames = $MaxFrames
  frame_count = $FrameCount
  overwrite = $Overwrite
  command = "$Python $($ArgsList -join ' ')"
}
$Meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $MetaFile -Encoding UTF8

Write-Output "STARTED native video frame cache build"
Write-Output "PID: $($Process.Id)"
Write-Output "RunId: $RunId"
Write-Output "FramesDir: $FramesDir"
Write-Output "CacheDir: $CacheDir"
Write-Output "Stdout: $StdoutLog"
Write-Output "Stderr: $StderrLog"
Write-Output "Meta: $MetaFile"

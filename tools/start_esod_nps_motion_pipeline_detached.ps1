param(
  [string]$Out = 'artifacts\esod_nps_motion_pipeline',
  [int]$Epochs = 10,
  [int]$BatchSize = 8,
  [int]$ImgSize = 640,
  [string]$Device = '0',
  [string]$TrainVideos = '1-40',
  [string]$ValVideos = '41-50',
  [int]$Stride = 4,
  [int]$MaxBoxes = 12,
  [int]$MaxFrames = 0
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$outAbs = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $repo $Out }
$python = Join-Path $repo 'papers\ESOD\.venv\Scripts\python.exe'
$runner = Join-Path $repo 'tools\run_esod_nps_motion_pipeline.py'
$logDir = Join-Path $outAbs 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stdout = Join-Path $logDir 'pipeline_runner.out.log'
$stderr = Join-Path $logDir 'pipeline_runner.err.log'
$pidFile = Join-Path $outAbs 'pipeline_pid.txt'
$metaFile = Join-Path $outAbs 'pipeline_meta.json'

$argsList = @(
  $runner,
  '--repo', $repo,
  '--out', $outAbs,
  '--epochs', "$Epochs",
  '--batch-size', "$BatchSize",
  '--img-size', "$ImgSize",
  '--device', $Device,
  '--train-videos', $TrainVideos,
  '--val-videos', $ValVideos,
  '--stride', "$Stride",
  '--max-boxes', "$MaxBoxes"
)
if ($MaxFrames -gt 0) { $argsList += @('--max-frames', "$MaxFrames") }

$proc = Start-Process -FilePath $python `
  -ArgumentList $argsList `
  -WorkingDirectory $repo `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

Set-Content -Path $pidFile -Value $proc.Id
@{
  start_time = (Get-Date).ToString('o')
  pid = $proc.Id
  stdout_log = $stdout
  stderr_log = $stderr
  output_root = $outAbs
  train_videos = $TrainVideos
  val_videos = $ValVideos
  epochs = $Epochs
  batch_size = $BatchSize
  img_size = $ImgSize
  device = $Device
  command = "$python $($argsList -join ' ')"
} | ConvertTo-Json -Depth 3 | Set-Content -Path $metaFile

Write-Host 'RUNNING'
Write-Host 'done/total: 0/2'
Write-Host ("pid: {0}" -f $proc.Id)
Write-Host ("start_time: {0}" -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))
Write-Host ("last_completed_unit: launched pipeline")
Write-Host ("stdout: {0}" -f $stdout)
Write-Host ("stderr: {0}" -f $stderr)
Write-Host ("output_root: {0}" -f $outAbs)

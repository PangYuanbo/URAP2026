param(
  [ValidateSet('nps','ard100')]
  [string]$Dataset = 'ard100',
  [string]$Videos = 'phantom09',
  [string]$Out = 'artifacts\motion_esod_rois',
  [int]$MaxFrames = 0,
  [int]$Stride = 4,
  [int]$MaxBoxes = 12,
  [switch]$SaveOverlays
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$python = Join-Path $repo 'papers\ESOD\.venv\Scripts\python.exe'
$script = Join-Path $repo 'tools\build_motion_esod_rois.py'
$outAbs = if ([System.IO.Path]::IsPathRooted($Out)) { $Out } else { Join-Path $repo $Out }
$logDir = Join-Path $outAbs 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logDir "motion_esod_${Dataset}_${stamp}.out.log"
$stderr = Join-Path $logDir "motion_esod_${Dataset}_${stamp}.err.log"
$pidFile = Join-Path $outAbs "motion_esod_${Dataset}_pid.txt"
$metaFile = Join-Path $outAbs "motion_esod_${Dataset}_meta.json"

$argsList = @(
  $script,
  '--dataset', $Dataset,
  '--videos', $Videos,
  '--out', $outAbs,
  '--stride', "$Stride",
  '--max-boxes', "$MaxBoxes"
)
if ($MaxFrames -gt 0) { $argsList += @('--max-frames', "$MaxFrames") }
if ($SaveOverlays) { $argsList += '--save-overlays' }

$proc = Start-Process -FilePath $python `
  -ArgumentList $argsList `
  -WorkingDirectory $repo `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

New-Item -ItemType Directory -Force -Path $outAbs | Out-Null
Set-Content -Path $pidFile -Value $proc.Id
@{
  dataset = $Dataset
  videos = $Videos
  start_time = (Get-Date).ToString('o')
  pid = $proc.Id
  stdout_log = $stdout
  stderr_log = $stderr
  output_root = $outAbs
  command = "$python $($argsList -join ' ')"
} | ConvertTo-Json -Depth 3 | Set-Content -Path $metaFile

Write-Host "RUNNING"
Write-Host ("done/total: 0/unknown")
Write-Host ("pid: {0}" -f $proc.Id)
Write-Host ("start_time: {0}" -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))
Write-Host ("stdout: {0}" -f $stdout)
Write-Host ("stderr: {0}" -f $stderr)
Write-Host ("output_root: {0}" -f $outAbs)

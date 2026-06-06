param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$ScriptPath = (Join-Path $RepoRoot 'tools\yolomg_timeline_eval.py'),
  [string]$Weights = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt',
  [string]$ImagesList = 'D:\URAP_datasets\ARD100_YOLOMG\test.txt',
  [string]$Images2List = 'D:\URAP_datasets\ARD100_YOLOMG\test2.txt',
  [string]$VideoRoot = 'D:\URAP_datasets\ARD100\test_videos',
  [string]$OutputDir = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\timeline_eval\test_timeline',
  [int]$ImgSize = 1280,
  [int]$BatchSize = 4,
  [string]$Device = '0',
  [int]$Workers = 4,
  [double]$WindowSeconds = 1.0,
  [string]$Metric = 'matched_confidence',
  [string[]]$VideoFilter = @(),
  [Nullable[int]]$StartFrame = $null,
  [Nullable[int]]$EndFrame = $null,
  [switch]$RenderOverlay,
  [switch]$SavePredLabels,
  [switch]$SavePredJsonl,
  [double]$OverlayAlpha = 0.88,
  [int]$PanelHeight = 220,
  [int]$LineWidth = 3,
  [string]$RunId = 'yolomg_timeline_eval',
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\detached_timeline_eval'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path -Path $ScriptPath -PathType Leaf)) { throw "ScriptPath not found: $ScriptPath" }
if (-not (Test-Path -Path $Weights -PathType Leaf)) { throw "Weights not found: $Weights" }
if (-not (Test-Path -Path $ImagesList -PathType Leaf)) { throw "ImagesList not found: $ImagesList" }
if (-not (Test-Path -Path $Images2List -PathType Leaf)) { throw "Images2List not found: $Images2List" }
if (-not (Test-Path -Path $VideoRoot -PathType Container)) { throw "VideoRoot not found: $VideoRoot" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $p = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
    if ($null -ne $p) {
      Write-Host "Timeline eval already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 120 }
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

$argList = @(
  $ScriptPath,
  '--weights', $Weights,
  '--images-list', $ImagesList,
  '--images2-list', $Images2List,
  '--video-root', $VideoRoot,
  '--output-dir', $OutputDir,
  '--imgsz', $ImgSize,
  '--batch-size', $BatchSize,
  '--device', $Device,
  '--workers', $Workers,
  '--window-seconds', $WindowSeconds,
  '--metric', $Metric,
  '--exist-ok'
)

if ($VideoFilter.Count -gt 0) {
  $argList += '--video-filter'
  $argList += $VideoFilter
}
if ($null -ne $StartFrame) {
  $argList += @('--start-frame', [string]$StartFrame)
}
if ($null -ne $EndFrame) {
  $argList += @('--end-frame', [string]$EndFrame)
}
if ($RenderOverlay) {
  $argList += '--render-overlay'
}
if ($SavePredLabels) {
  $argList += '--save-pred-labels'
}
if ($SavePredJsonl) {
  $argList += '--save-pred-jsonl'
}
if ($OverlayAlpha -ne 0.88) {
  $argList += @('--overlay-alpha', [string]$OverlayAlpha)
}
if ($PanelHeight -ne 220) {
  $argList += @('--panel-height', [string]$PanelHeight)
}
if ($LineWidth -ne 3) {
  $argList += @('--line-width', [string]$LineWidth)
}

$p = Start-Process `
  -FilePath $PythonExe `
  -ArgumentList $argList `
  -WorkingDirectory $RepoRoot `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

$p.Id | Set-Content -Encoding ascii -Path $pidFile

@(
  ('started={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
  ('pid={0}' -f $p.Id),
  ('python={0}' -f $PythonExe),
  ('script={0}' -f $ScriptPath),
  ('run_id={0}' -f $RunId),
  ('output_dir={0}' -f $OutputDir),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('cmd_args={0}' -f ($argList -join ' '))
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached YOLOMG timeline evaluation.'
Get-Content $metaFile

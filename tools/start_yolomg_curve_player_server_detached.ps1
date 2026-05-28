param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'papers\ESOD\.venv\Scripts\python.exe'),
  [string]$PlayerHtml = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\timeline_eval\test_timeline_dualcharts_20260323\yolomg_video_curve_player.html',
  [string]$VideoRoot = 'D:\URAP_datasets\ARD100\test_videos',
  [int]$Port = 8777,
  [string]$RunId = 'yolomg_curve_player_server',
  [string]$OutputRoot = 'C:\Users\aaron\Desktop\URAP\artifacts\yolomg_curve_player_server'
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path -Path $PlayerHtml -PathType Leaf)) { throw "PlayerHtml not found: $PlayerHtml" }
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
      Write-Host "YOLOMG curve player server already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 120 }
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)
$scriptPath = Join-Path $RepoRoot 'tools\yolomg_curve_player_server.py'

$argList = @(
  $scriptPath,
  '--player-html', $PlayerHtml,
  '--video-root', $VideoRoot,
  '--host', '127.0.0.1',
  '--port', $Port
)

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
  ('url=http://127.0.0.1:{0}/' -f $Port),
  ('python={0}' -f $PythonExe),
  ('script={0}' -f $scriptPath),
  ('run_id={0}' -f $RunId),
  ('player_html={0}' -f $PlayerHtml),
  ('video_root={0}' -f $VideoRoot),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr)
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached YOLOMG curve player server.'
Get-Content $metaFile

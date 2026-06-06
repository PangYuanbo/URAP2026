param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'),
  [string]$ScriptPath = (Join-Path $RepoRoot 'tools\yolomg_eval_pred_labels.py'),
  [string]$ImagesList = 'D:\URAP_datasets\ARD100_YOLOMG\test.txt',
  [string]$PredLabelDir,
  [string]$OutDir,
  [double]$ConfThres = 0.001,
  [double]$MatchIou = 0.5,
  [Nullable[int]]$ImageWidth = 1920,
  [Nullable[int]]$ImageHeight = 1080,
  [string]$RunId = 'yolomg_pred_label_eval',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\yolomg_pred_label_eval_runner')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path -Path $ScriptPath -PathType Leaf)) { throw "ScriptPath not found: $ScriptPath" }
if (-not (Test-Path -Path $ImagesList -PathType Leaf)) { throw "ImagesList not found: $ImagesList" }
if (-not (Test-Path -Path $PredLabelDir -PathType Container)) { throw "PredLabelDir not found: $PredLabelDir" }
if (-not $OutDir) { throw 'OutDir must be provided' }
if (($null -eq $ImageWidth) -ne ($null -eq $ImageHeight)) { throw 'ImageWidth and ImageHeight must be provided together' }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)
if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*yolomg_eval_pred_labels.py*') {
      Write-Host "YOLOMG pred-label eval already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)
$argList = @(
  $ScriptPath,
  '--images-list', $ImagesList,
  '--pred-label-dir', $PredLabelDir,
  '--out-dir', $OutDir,
  '--conf-thres', [string]$ConfThres,
  '--match-iou', [string]$MatchIou
)
if ($null -ne $ImageWidth) {
  $argList += @('--image-width', [string]$ImageWidth, '--image-height', [string]$ImageHeight)
}

$env:PYTHONPATH = $RepoRoot
$proc = Start-Process -FilePath $PythonExe -ArgumentList $argList -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$proc.Id | Set-Content -Encoding ascii -Path $pidFile
@(
  ('started={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
  ('pid={0}' -f $proc.Id),
  ('python={0}' -f $PythonExe),
  ('script={0}' -f $ScriptPath),
  ('run_id={0}' -f $RunId),
  ('images_list={0}' -f $ImagesList),
  ('pred_label_dir={0}' -f $PredLabelDir),
  ('out_dir={0}' -f $OutDir),
  ('conf_thres={0}' -f $ConfThres),
  ('match_iou={0}' -f $MatchIou),
  ('image_width={0}' -f $ImageWidth),
  ('image_height={0}' -f $ImageHeight),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('cmd_args={0}' -f ($argList -join ' '))
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached YOLOMG pred-label evaluation.'
Get-Content $metaFile

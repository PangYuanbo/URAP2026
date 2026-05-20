param(
  [string]$RepoDir = "C:\Users\aaron\Desktop\URAP\papers\ESOD",
  [string]$PythonExe = "C:\Users\aaron\Desktop\URAP\papers\ESOD\.venv\Scripts\python.exe",
  [string]$Project = "runs/train",
  [string]$Cfg = "models/cfg/esod/visdrone_yolov5m.yaml",
  [string]$Data = "data/visdrone.yaml",
  [string]$Hyp = "data/hyps/hyp.visdrone.yaml",
  [string]$Weights = "weights/pretrained/yolov5m.pt",
  [int]$Epochs = 50,
  [int]$BatchSize = 8,
  [int]$ImgSize = 1536,
  [string]$Device = "0",
  [int]$Workers = 0,
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\ESOD\runs\train_visdrone_yolov5m_detached",
  [string]$RunId = "visdrone_yolov5m_e50"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $RepoDir -PathType Container)) { throw "RepoDir not found: $RepoDir" }
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

# Prevent duplicate concurrent runs.
if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $p = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
    if ($null -ne $p) {
      Write-Host "Already running: pid=$existingPid"
      if (Test-Path $metaFile) { Get-Content $metaFile | Select-Object -First 120 }
      exit 0
    }
  }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$metaBackup = Join-Path $OutputRoot ("runner_{0}_meta_prev_{1}.txt" -f $RunId, $ts)
$pidBackup = Join-Path $OutputRoot ("runner_{0}_pid_prev_{1}.txt" -f $RunId, $ts)

if (Test-Path -Path $metaFile -PathType Leaf) {
  Copy-Item -Force -Path $metaFile -Destination $metaBackup
}
if (Test-Path -Path $pidFile -PathType Leaf) {
  Copy-Item -Force -Path $pidFile -Destination $pidBackup
}

$runName = ("visdrone_esod_yolov5m_e{0}_b{1}_img{2}_{3}" -f $Epochs, $BatchSize, $ImgSize, $ts)

$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)

# Avoid W&B login prompts or network calls.
$env:WANDB_MODE = "disabled"
$env:WANDB_DISABLED = "true"

$args = @(
  ".\train.py",
  "--data", $Data,
  "--cfg", $Cfg,
  "--weights", $Weights,
  "--hyp", $Hyp,
  "--batch-size", $BatchSize,
  "--img-size", $ImgSize,
  "--epochs", $Epochs,
  "--device", $Device,
  "--workers", $Workers,
  "--project", $Project,
  "--name", $runName
)

$p = Start-Process `
  -FilePath $PythonExe `
  -ArgumentList $args `
  -WorkingDirectory $RepoDir `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

$p.Id | Set-Content -Encoding ascii -Path $pidFile

$saveDir = Join-Path (Join-Path $RepoDir $Project) $runName

@(
  ("started={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
  ("pid={0}" -f $p.Id)
  ("repo_dir={0}" -f $RepoDir)
  ("python={0}" -f $PythonExe)
  ("run_id={0}" -f $RunId)
  ("run_name={0}" -f $runName)
  ("save_dir={0}" -f $saveDir)
  ("stdout={0}" -f $stdout)
  ("stderr={0}" -f $stderr)
  ("cmd_args={0}" -f ($args -join " "))
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host "Started detached ESOD training runner."
Get-Content $metaFile

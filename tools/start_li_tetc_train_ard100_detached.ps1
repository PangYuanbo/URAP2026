param(
  [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking",
  [string]$ProjectRoot = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\baselines\li_tetc_pt_pipeline",
  [string]$ARDRoot = "D:\URAP_datasets\ARD100",
  [string]$ARDSplitRoot = "D:\URAP_datasets\TransVisDrone\ARD100\Videos",
  [string[]]$Videos = @(),
  [int]$Epochs = 5,
  [int]$BatchSize = 2,
  [int]$FrameStride = 3,
  [int]$MaxFramesPerVideo = 0,
  [int]$EmptyStride = 10,
  [double]$LR = 0.002,
  [int]$MinSize = 1080,
  [int]$MaxSize = 1920,
  [string]$AnchorPreset = "tiny",
  [string]$RunId = "",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\baselines\li_tetc_pt_pipeline\runs\detached"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -Path $ProjectRoot -PathType Container)) { throw "ProjectRoot not found: $ProjectRoot" }
$pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -Path $pythonExe -PathType Leaf)) { throw "Python exe not found: $pythonExe" }
$helperScript = "C:\Users\aaron\Desktop\URAP\tools\print_ard100_split_ids.py"
if (-not (Test-Path -Path $helperScript -PathType Leaf)) { throw "Split helper script not found: $helperScript" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

if (-not $RunId) { $RunId = "ard100_train_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
  if ($existingPid -match '^\d+$') {
    $p = Get-Process -Id ([int]$existingPid) -ErrorAction SilentlyContinue
    if ($null -ne $p) { throw "Runner already active with pid=$existingPid. Stop/inspect existing run first." }
  }
}

if ($Videos.Count -eq 0) {
  $splitIds = & $pythonExe $helperScript --split-root $ARDSplitRoot --split train
  if ($LASTEXITCODE -ne 0) { throw "Failed to load ARD100 train split ids." }
  $Videos = @($splitIds -split '\s+' | Where-Object { $_ })
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$ckpt = Join-Path $OutputRoot ("fasterrcnn_ard100_{0}.pt" -f $RunId)
$stdout = Join-Path $logsDir ("runner_train_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_train_{0}_{1}.err.txt" -f $RunId, $ts)

$argList = @(
  "train_detector.py",
  "--repo-root", $RepoRoot,
  "--dataset", "ard100",
  "--ard-root", $ARDRoot,
  "--ard-split-root", $ARDSplitRoot,
  "--frame-stride", $FrameStride,
  "--include-empty",
  "--empty-stride", $EmptyStride,
  "--epochs", $Epochs,
  "--batch-size", $BatchSize,
  "--lr", $LR,
  "--min-size", $MinSize,
  "--max-size", $MaxSize,
  "--anchor-preset", $AnchorPreset,
  "--out", $ckpt,
  "--amp",
  "--videos"
)
$argList += $Videos
if ($MaxFramesPerVideo -gt 0) { $argList += @("--max-frames", $MaxFramesPerVideo) }

$p = Start-Process -FilePath $pythonExe -ArgumentList $argList -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$p.Id | Set-Content -Encoding Ascii -Path $pidFile

@(
  ("started={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
  ("pid={0}" -f $p.Id),
  ("repo_root={0}" -f $RepoRoot),
  ("project_root={0}" -f $ProjectRoot),
  ("python={0}" -f $pythonExe),
  ("run_id={0}" -f $RunId),
  ("ckpt={0}" -f $ckpt),
  ("stdout={0}" -f $stdout),
  ("stderr={0}" -f $stderr),
  ("argv={0}" -f ($argList -join " ")),
  ("total_epochs={0}" -f $Epochs),
  ("dataset=ard100"),
  ("videos={0}" -f ($Videos -join ",")),
  ("ard_root={0}" -f $ARDRoot),
  ("ard_split_root={0}" -f $ARDSplitRoot)
) | Set-Content -Encoding Ascii -Path $metaFile

Write-Host ("Started LI TETC ARD100 train detached: pid={0}, run_id={1}" -f $p.Id, $RunId)
Write-Host ("meta: {0}" -f $metaFile)
Get-Content $metaFile

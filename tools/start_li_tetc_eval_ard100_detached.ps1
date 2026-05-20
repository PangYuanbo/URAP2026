param(
  [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking",
  [string]$ProjectRoot = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\baselines\li_tetc_pt_pipeline",
  [string]$ARDRoot = "D:\URAP_datasets\ARD100",
  [string]$ARDSplitRoot = "D:\URAP_datasets\TransVisDrone\ARD100\Videos",
  [ValidateSet("val", "test")]
  [string]$Split = "val",
  [string]$Checkpoint,
  [string[]]$Videos = @(),
  [int]$FrameStride = 3,
  [int]$MaxFramesPerVideo = 0,
  [int]$EmptyStride = 10,
  [int]$BatchSize = 1,
  [int]$NumWorkers = 4,
  [int]$MinSize = 1080,
  [int]$MaxSize = 1920,
  [string[]]$Scores = @("0.1", "0.2", "0.3", "0.4", "0.5"),
  [double]$Iou = 0.5,
  [string]$RunId = "",
  [string]$OutJson = "",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\baselines\li_tetc_pt_pipeline\runs\detached"
)

$ErrorActionPreference = "Stop"
if (-not $Checkpoint) { throw "Checkpoint is required." }
if (-not (Test-Path -Path $Checkpoint -PathType Leaf)) { throw "Checkpoint not found: $Checkpoint" }
if (-not (Test-Path -Path $ProjectRoot -PathType Container)) { throw "ProjectRoot not found: $ProjectRoot" }
$pythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -Path $pythonExe -PathType Leaf)) { throw "Python exe not found: $pythonExe" }
$helperScript = "C:\Users\aaron\Desktop\URAP\tools\print_ard100_split_ids.py"
if (-not (Test-Path -Path $helperScript -PathType Leaf)) { throw "Split helper script not found: $helperScript" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

if (-not $RunId) { $RunId = "ard100_eval_${Split}_$(Get-Date -Format 'yyyyMMdd_HHmmss')" }
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
  $splitIds = & $pythonExe $helperScript --split-root $ARDSplitRoot --split $Split
  if ($LASTEXITCODE -ne 0) { throw "Failed to load ARD100 $Split split ids." }
  $Videos = @($splitIds -split '\s+' | Where-Object { $_ })
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
if (-not $OutJson) { $OutJson = Join-Path $OutputRoot ("eval_{0}_{1}.json" -f $RunId, $Split) }
$stdout = Join-Path $logsDir ("runner_eval_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_eval_{0}_{1}.err.txt" -f $RunId, $ts)

$argList = @(
  "-u",
  "eval_detector.py",
  "--repo-root", $RepoRoot,
  "--dataset", "ard100",
  "--ard-root", $ARDRoot,
  "--ard-split-root", $ARDSplitRoot,
  "--ard-split", $Split,
  "--ckpt", $Checkpoint,
  "--frame-stride", $FrameStride,
  "--include-empty",
  "--empty-stride", $EmptyStride,
  "--batch-size", $BatchSize,
  "--num-workers", $NumWorkers,
  "--min-size", $MinSize,
  "--max-size", $MaxSize,
  "--iou", $Iou,
  "--out-json", $OutJson
)
if ($MaxFramesPerVideo -gt 0) { $argList += @("--max-frames", $MaxFramesPerVideo) }
foreach ($s in $Scores) { $argList += @("--scores", $s) }
$argList += "--videos"
$argList += $Videos

$p = Start-Process -FilePath $pythonExe -ArgumentList $argList -WorkingDirectory $ProjectRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$p.Id | Set-Content -Encoding Ascii -Path $pidFile

@(
  ("started={0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss")),
  ("pid={0}" -f $p.Id),
  ("repo_root={0}" -f $RepoRoot),
  ("project_root={0}" -f $ProjectRoot),
  ("python={0}" -f $pythonExe),
  ("run_id={0}" -f $RunId),
  ("split={0}" -f $Split),
  ("ckpt={0}" -f $Checkpoint),
  ("stdout={0}" -f $stdout),
  ("stderr={0}" -f $stderr),
  ("out_json={0}" -f $OutJson),
  ("argv={0}" -f ($argList -join " ")),
  ("dataset=ard100"),
  ("videos={0}" -f ($Videos -join ",")),
  ("ard_root={0}" -f $ARDRoot),
  ("ard_split_root={0}" -f $ARDSplitRoot)
) | Set-Content -Encoding Ascii -Path $metaFile

Write-Host ("Started LI TETC ARD100 eval detached: pid={0}, run_id={1}" -f $p.Id, $RunId)
Write-Host ("meta: {0}" -f $metaFile)
Get-Content $metaFile

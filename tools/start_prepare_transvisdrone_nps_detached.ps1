param(
  [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$Python = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe",
  [string]$VideosDir = "C:\Users\aaron\Desktop\URAP\Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking\Data\Videos",
  [string]$AnnosDir = "C:\Users\aaron\Desktop\URAP\datasets\Drone-Detection\annotations\NPS-Drones-Dataset",
  [string]$OutRoot = "D:\URAP_datasets\TransVisDrone\NPS",
  [ValidateSet("train", "val", "test")]
  [string]$OnlySplit = "train",
  [int]$PngCompression = 3,
  [string]$RunId = "prepare_transvisdrone_nps",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_sota_research\prepare_transvisdrone_nps_runner"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -Path $Python -PathType Leaf)) { throw "Python not found: $Python" }
if (-not (Test-Path -Path $VideosDir -PathType Container)) { throw "VideosDir not found: $VideosDir" }
if (-not (Test-Path -Path $AnnosDir -PathType Container)) { throw "AnnosDir not found: $AnnosDir" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

if (Test-Path $pidFile) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid) {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like "*prepare_transvisdrone_nps.py*" -and $existing.CommandLine -like "*--only-split $OnlySplit*") {
      Write-Host "Prepare TransVisDrone NPS already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logsDir "runner_${RunId}_${ts}.out.txt"
$stderr = Join-Path $logsDir "runner_${RunId}_${ts}.err.txt"
$argList = @(
  "tools\prepare_transvisdrone_nps.py",
  "--videos-dir", $VideosDir,
  "--annos-dir", $AnnosDir,
  "--out-root", $OutRoot,
  "--png-compression", [string]$PngCompression,
  "--only-split", $OnlySplit
)

$proc = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -WindowStyle Hidden
$proc.Id | Set-Content -Path $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($proc.Id)",
  "python=$Python",
  "run_id=$RunId",
  "repo_root=$RepoRoot",
  "videos_dir=$VideosDir",
  "annos_dir=$AnnosDir",
  "out_root=$OutRoot",
  "only_split=$OnlySplit",
  "png_compression=$PngCompression",
  "output_root=$OutputRoot",
  "stdout=$stdout",
  "stderr=$stderr",
  "cmd_args=$($argList -join ' ')"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Host "Started detached TransVisDrone NPS prepare."
Get-Content $metaFile

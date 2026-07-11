param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$PythonExe = (Join-Path $RepoRoot 'papers\TransVisDrone\.venv\Scripts\python.exe'),
  [string]$OutputDir = 'U:\URAP_datasets\FL-Drones',
  [string]$RunId = 'fl_drones_official_download',
  [string]$OutputRoot = (Join-Path $RepoRoot 'artifacts\benchmarks\fl_drones_download')
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python not found: $PythonExe" }
New-Item -ItemType Directory -Force -Path $OutputDir, $OutputRoot | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$pidFile = Join-Path $OutputRoot ("{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("{0}_meta.txt" -f $RunId)
if (Test-Path -LiteralPath $pidFile) {
  $existingPid = Get-Content -LiteralPath $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*gdown*18CoTpjMs80dfanYNpbznjL4e-KB_Diel*') {
      Write-Host "FL-Drones download already running: pid=$existingPid"
      exit 0
    }
  }
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("{0}_{1}.out.txt" -f $RunId, $timestamp)
$stderr = Join-Path $logsDir ("{0}_{1}.err.txt" -f $RunId, $timestamp)
$outputFile = Join-Path $OutputDir 'uav200.zip'
$arguments = @(
  '-m', 'gdown',
  '--fuzzy',
  'https://drive.google.com/open?id=18CoTpjMs80dfanYNpbznjL4e-KB_Diel',
  '-O', $outputFile
)

$process = Start-Process -FilePath $PythonExe -ArgumentList $arguments -WorkingDirectory $RepoRoot `
  -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$process.Id | Set-Content -LiteralPath $pidFile -Encoding ascii
@(
  "started=$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))",
  "pid=$($process.Id)",
  "python=$PythonExe",
  "output_dir=$OutputDir",
  "output_file=$outputFile",
  "stdout=$stdout",
  "stderr=$stderr",
  "source=https://www.epfl.ch/labs/cvlab/research/uav/research-unmanned-detection/",
  "drive_folder=18CoTpjMs80dfanYNpbznjL4e-KB_Diel"
) | Set-Content -LiteralPath $metaFile -Encoding utf8

Write-Host 'Started detached FL-Drones official download.'
Get-Content -LiteralPath $metaFile

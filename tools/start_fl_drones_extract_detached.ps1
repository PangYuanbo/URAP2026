param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$Archive = 'U:\URAP_datasets\FL-Drones\uav200.zip',
  [string]$Destination = 'U:\URAP_datasets\FL-Drones\uav200',
  [string]$PythonExe = '',
  [string]$RunId = 'fl_drones_extract',
  [string]$OutputRoot = ''
)

$ErrorActionPreference = 'Stop'
if (-not $PythonExe) { $PythonExe = Join-Path $RepoRoot 'papers\TransVisDrone\.venv\Scripts\python.exe' }
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot 'artifacts\benchmarks\fl_drones_extract' }
$Archive = [System.IO.Path]::GetFullPath($Archive)
$Destination = [System.IO.Path]::GetFullPath($Destination)
if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) { throw "Archive not found: $Archive" }
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python not found: $PythonExe" }
New-Item -ItemType Directory -Force -Path $Destination, $OutputRoot, (Join-Path $OutputRoot 'logs') | Out-Null
$pidFile = Join-Path $OutputRoot ("{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("{0}_meta.txt" -f $RunId)
$progressFile = Join-Path $OutputRoot ("{0}_progress.json" -f $RunId)
if (Test-Path -LiteralPath $pidFile) {
  $existingPid = Get-Content -LiteralPath $pidFile | Select-Object -First 1
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*extract_zip_with_progress.py*') { Write-Host "Already running PID=$existingPid"; exit 0 }
  }
}
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $OutputRoot ("logs\{0}_{1}.out.txt" -f $RunId, $timestamp)
$stderr = Join-Path $OutputRoot ("logs\{0}_{1}.err.txt" -f $RunId, $timestamp)
$arguments = @(
  (Join-Path $RepoRoot 'tools\extract_zip_with_progress.py'),
  '--archive', $Archive,
  '--destination', $Destination,
  '--progress', $progressFile
)
$process = Start-Process -FilePath $PythonExe -ArgumentList $arguments -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$process.Id | Set-Content -LiteralPath $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($process.Id)",
  "archive=$Archive",
  "destination=$Destination",
  "progress=$progressFile",
  "stdout=$stdout",
  "stderr=$stderr"
) | Set-Content -LiteralPath $metaFile -Encoding ascii
Get-Content -LiteralPath $metaFile

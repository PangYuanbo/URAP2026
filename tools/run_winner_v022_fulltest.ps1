param(
  [string]$DatasetPath = "D:\URAP_datasets\AOT\part1\Images",
  [string]$FlightIdsJson = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\aot_flight_ids\testflightidsfull1.json",
  [string]$RepoDir = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\submission-v022\airborne-detection-starter-kit-submission-v022",
  [string]$PythonExe = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python.exe",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_fulltest",
  [string]$RunId = "fulltest"
)

$ErrorActionPreference = "Stop"
Write-Warning "This runner executes in the current PowerShell session. For a detached/background run (recommended), use tools\\start_winner_v022_fulltest_detached.ps1 and tools\\monitor_winner_v022_fulltest.ps1."

if (-not (Test-Path -Path $DatasetPath -PathType Container)) {
  throw "DatasetPath not found: $DatasetPath"
}
if (-not (Test-Path -Path $FlightIdsJson -PathType Leaf)) {
  throw "FlightIdsJson not found: $FlightIdsJson"
}
if (-not (Test-Path -Path $RepoDir -PathType Container)) {
  throw "RepoDir not found: $RepoDir"
}
if (-not (Test-Path -Path $PythonExe -PathType Leaf)) {
  throw "PythonExe not found: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$runOut = Join-Path $OutputRoot $RunId
New-Item -ItemType Directory -Force -Path $runOut | Out-Null

$ids = Get-Content $FlightIdsJson | ConvertFrom-Json
if ($null -eq $ids -or $ids.Count -eq 0) {
  throw "No flight ids loaded from $FlightIdsJson"
}

$done = @{}
Get-ChildItem -Path $runOut -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  $rid = Join-Path $_.FullName "result.json"
  if (Test-Path -Path $rid -PathType Leaf) {
    $done[$_.Name] = $true
  }
}

$remaining = @()
foreach ($id in $ids) {
  if (-not $done.ContainsKey($id)) {
    $remaining += $id
  }
}

Write-Host ("Total test flights: {0}" -f $ids.Count)
Write-Host ("Already done:       {0}" -f $done.Keys.Count)
Write-Host ("Remaining:          {0}" -f $remaining.Count)

if ($remaining.Count -eq 0) {
  Write-Host "Nothing to do."
  exit 0
}

Set-Location $RepoDir
$env:TEST_DATASET_PATH = $DatasetPath
$env:INFERENCE_OUTPUT_PATH = $OutputRoot
$env:DATASET_ENV = $RunId
$env:PARTIAL_RUN_FLIGHTS = ($remaining -join ",")

& $PythonExe .\seg_test.py
if ($LASTEXITCODE -ne 0) {
  throw "seg_test.py failed with exit code $LASTEXITCODE"
}

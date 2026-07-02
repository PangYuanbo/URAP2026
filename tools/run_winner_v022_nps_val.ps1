param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$DatasetPath = "D:\URAP_datasets\TransVisDrone\NPS\AllFrames\val",
  [string]$RepoDir = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\submission-v022\airborne-detection-starter-kit-submission-v022",
  [string]$PythonExe = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python.exe",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val",
  [string]$RunId = "nps_val",
  [string]$PreparedDatasetPath = "",
  [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $URAPRoot -PathType Container)) {
  throw "URAPRoot not found: $URAPRoot"
}
if (-not (Test-Path -Path $DatasetPath -PathType Container)) {
  throw "DatasetPath not found: $DatasetPath"
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

if (-not $PreparedDatasetPath) {
  $PreparedDatasetPath = Join-Path $OutputRoot ("_prepared_{0}" -f $RunId)
}
$testDatasetPath = $DatasetPath

if (-not $SkipPrepare) {
  $prepScript = Join-Path $URAPRoot "tools\prepare_aicrowd_nps_flight_dirs.py"
  if (-not (Test-Path -Path $prepScript -PathType Leaf)) { throw "prepare_aicrowd_nps_flight_dirs.py not found: $prepScript" }
  $prepReport = Join-Path $OutputRoot ("prepare_{0}.json" -f $RunId)
  & $PythonExe $prepScript `
    --frames-dir $DatasetPath `
    --out-dir $PreparedDatasetPath `
    --json $prepReport
  if ($LASTEXITCODE -ne 0) { throw "prepare_aicrowd_nps_flight_dirs.py failed with exit code $LASTEXITCODE" }
  $testDatasetPath = $PreparedDatasetPath
}

# Enumerate clip ids from AICrowd-style folder layout.
$allClips = @(Get-ChildItem -Path $testDatasetPath -Directory -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty Name |
  Sort-Object -Unique)

if ($allClips.Count -eq 0) {
  throw "No NPS clip folders found under: $testDatasetPath"
}

# Resume logic: skip clips whose per-clip result.json already exists.
$done = @{}
Get-ChildItem -Path $runOut -Directory -ErrorAction SilentlyContinue | ForEach-Object {
  $rid = Join-Path $_.FullName "result.json"
  if (Test-Path -Path $rid -PathType Leaf) {
    $done[$_.Name] = $true
  }
}

$remaining = @()
foreach ($clip in $allClips) {
  if (-not $done.ContainsKey($clip)) {
    $remaining += $clip
  }
}

Write-Host ("NPS val clips:       {0}" -f $allClips.Count)
Write-Host ("Already done:        {0}" -f $done.Keys.Count)
Write-Host ("Remaining:           {0}" -f $remaining.Count)

if ($remaining.Count -eq 0) {
  Write-Host "Nothing to do."
  exit 0
}

Set-Location $RepoDir
$env:TEST_DATASET_PATH = $testDatasetPath
$env:INFERENCE_OUTPUT_PATH = $OutputRoot
$env:DATASET_ENV = $RunId
$env:PARTIAL_RUN_FLIGHTS = ($remaining -join ",")

& $PythonExe .\seg_test.py
if ($LASTEXITCODE -ne 0) {
  throw "seg_test.py failed with exit code $LASTEXITCODE"
}

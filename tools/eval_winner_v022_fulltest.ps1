param(
  [string]$PythonExe = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python.exe",
  [string]$WorkspaceRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$TransVisDroneDir = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone",
  [string]$DatasetFolder = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\runs\eval\AOT_URAP\fulltest_conf0p2\gt",
  [string]$ResultsFolder = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_fulltest\fulltest",
  [string]$SummariesFolder = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\compare_fulltest\winner_v022\summaries",
  [int]$MinFlights = 172,
  [double]$MinScore = 0.2,
  [int]$MinTrackLen = 0,
  [int]$EncMaxRange = 700
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path -Path $WorkspaceRoot -PathType Container)) { throw "WorkspaceRoot not found: $WorkspaceRoot" }
if (-not (Test-Path -Path $TransVisDroneDir -PathType Container)) { throw "TransVisDroneDir not found: $TransVisDroneDir" }
if (-not (Test-Path -Path $DatasetFolder -PathType Container)) { throw "DatasetFolder not found: $DatasetFolder" }
if (-not (Test-Path -Path $ResultsFolder -PathType Container)) { throw "ResultsFolder not found: $ResultsFolder" }

New-Item -ItemType Directory -Force -Path $SummariesFolder | Out-Null

# Merge per-flight results into a single result.json expected by airborne metrics.
Set-Location $WorkspaceRoot
& $PythonExe tools\merge_airborne_results.py --results-dir $ResultsFolder --min-flights $MinFlights --sort
if ($LASTEXITCODE -ne 0) { throw "merge_airborne_results.py failed with exit code $LASTEXITCODE" }

Set-Location $TransVisDroneDir
& $PythonExe -m aotcore.metrics.run_airborne_metrics `
  --dataset-folder $DatasetFolder `
  --results-folder $ResultsFolder `
  --summaries-folder $SummariesFolder `
  --min-score $MinScore `
  --min-track-len $MinTrackLen `
  --enc-max-range $EncMaxRange

if ($LASTEXITCODE -ne 0) { throw "run_airborne_metrics failed with exit code $LASTEXITCODE" }

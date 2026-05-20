param(
  [string]$PythonExe = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\.venv\Scripts\python.exe",
  [string]$ImagesDir = "D:\URAP_datasets\TransVisDrone\NPS\AllFrames\val",
  [string]$LabelsDir = "D:\URAP_datasets\TransVisDrone\NPS\NPSvisdroneStyle\val\labels",
  [string]$ResultsFolder = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_nps_val\nps_val",
  [string]$SummariesFolder = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\compare_nps_val\winner_v022\summaries",
  [double]$MinScore = 0.0,
  [double]$IouThr = 0.5,
  [int]$ImgW = 1280,
  [int]$ImgH = 960
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -Path $PythonExe -PathType Leaf)) { throw "PythonExe not found: $PythonExe" }
if (-not (Test-Path -Path $ImagesDir -PathType Container)) { throw "ImagesDir not found: $ImagesDir" }
if (-not (Test-Path -Path $LabelsDir -PathType Container)) { throw "LabelsDir not found: $LabelsDir" }
if (-not (Test-Path -Path $ResultsFolder -PathType Container)) { throw "ResultsFolder not found: $ResultsFolder" }

New-Item -ItemType Directory -Force -Path $SummariesFolder | Out-Null

# Merge per-clip outputs (Clip_xx/result.json) to a single result.json for stable inspection.
Set-Location "C:\Users\aaron\Desktop\URAP"
& $PythonExe tools\merge_airborne_results.py --results-dir $ResultsFolder --sort
if ($LASTEXITCODE -ne 0) { throw "merge_airborne_results.py failed with exit code $LASTEXITCODE" }

$outJson = Join-Path $SummariesFolder "winner_v022_nps_val_ap_iou${IouThr}_minScore${MinScore}.json"
& $PythonExe tools\eval_winner_v022_nps_val.py `
  --images-dir $ImagesDir `
  --labels-dir $LabelsDir `
  --results-dir $ResultsFolder `
  --img-w $ImgW `
  --img-h $ImgH `
  --iou-thr $IouThr `
  --min-score $MinScore `
  --out-json $outJson

if ($LASTEXITCODE -ne 0) { throw "eval_winner_v022_nps_val.py failed with exit code $LASTEXITCODE" }


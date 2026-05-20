param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$AOTRoot = "D:\URAP_datasets\AOT\part1",
  [string]$OutRoot = "D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest",
  [int]$PartSize = 10,
  [int]$DownloadParallel = 16,
  [int]$BatchSize = 2,
  [int]$Img = 1280,
  [int]$NumFrames = 3,
  [double]$ConfThres = 0.2,
  [string]$RunNameSuffix = "",
  [string[]]$ExtraValArgs = @(),
  [switch]$SkipPrepare,
  [switch]$SkipInfer,
  [switch]$SkipEval
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Ensure-Dirs([string[]]$dirs) {
  foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
  }
}

function Invoke-Native([string]$Exe, [string[]]$ArgList, [string]$LogPath) {
  # Windows PowerShell treats native STDERR as non-terminating errors. Many Python loggers
  # write INFO logs to STDERR, which would abort the script if $ErrorActionPreference=Stop.
  # We rely on $LASTEXITCODE instead.
  $oldEap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & $Exe @ArgList 2>&1 | Tee-Object -FilePath $LogPath
    $code = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $oldEap
  }
  if ($code -ne 0) {
    throw ("Native command failed with exit code {0}: {1} {2}" -f $code, $Exe, ($ArgList -join " "))
  }
}

$repo = Join-Path $URAPRoot "papers\TransVisDrone"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$weights = Join-Path $repo "pretrained\TransVisDrone_weights\runs\train\AOT\image_size_1280_YOLOXL_3_frames_AOT_with_yolo_weights_end_to_end\weights\best.pt"
$yamlDir = Join-Path $repo "data\AOTTestSplits_URAP"

if (!(Test-Path $py)) { throw "Missing python venv: $py" }
if (!(Test-Path $weights)) { throw "Missing weights: $weights" }

$runName = "fulltest_conf{0}" -f ($ConfThres.ToString().Replace(".", "p"))
if ($RunNameSuffix) {
  $runName = ("{0}_{1}" -f $runName, $RunNameSuffix)
}
$project = Join-Path $repo "runs\val\AOT_URAP"
$evalOut = Join-Path $repo "runs\eval\AOT_URAP\$runName"
$logDir = Join-Path $URAPRoot ("artifacts\\logs\\aot_fulltest\\{0}" -f $runName)
Ensure-Dirs @($logDir)

# TransVisDrone's dataset checker expects these folders to exist (even if empty for test-only runs).
Ensure-Dirs @(
  (Join-Path $OutRoot "train\frames"),
  (Join-Path $OutRoot "train\labels"),
  (Join-Path $OutRoot "train\videos"),
  (Join-Path $OutRoot "val\full\frames"),
  (Join-Path $OutRoot "val\full\labels"),
  (Join-Path $OutRoot "val\full\videos")
)

Write-Host ("[{0}] AOT full-test pipeline starting" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host ("Repo: {0}" -f $repo)
Write-Host ("AOT root: {0}" -f $AOTRoot)
Write-Host ("Out root: {0}" -f $OutRoot)
Write-Host ("Run name: {0}" -f $runName)

if (-not $SkipPrepare) {
  $prepareLog = Join-Path $logDir "prepare.log"
  Write-Host ("[{0}] Step 1/3: prepare dataset -> {1}" -f (Get-Date -Format "HH:mm:ss"), $prepareLog)
  Invoke-Native $py @(
    (Join-Path $URAPRoot "tools\prepare_transvisdrone_aot_part1.py"),
    "--aot-root", $AOTRoot,
    "--out-root", $OutRoot,
    "--split", "test",
    "--part-size", $PartSize,
    "--download-parallel", $DownloadParallel
  ) $prepareLog
}

if (-not $SkipInfer) {
  $yamlFiles = @(Get-ChildItem -Path $yamlDir -Filter "AOTTest_*.yaml" | Sort-Object Name)
  if ($yamlFiles.Count -eq 0) { throw "No YAMLs found in: $yamlDir (prepare step may have failed)" }

  Write-Host ("[{0}] Step 2/3: inference over {1} parts" -f (Get-Date -Format "HH:mm:ss"), $yamlFiles.Count)
  foreach ($yf in $yamlFiles) {
    $splitNum = [System.IO.Path]::GetFileNameWithoutExtension($yf.Name).Split("_")[-1]
    $predPath = Join-Path $project "$runName\aotpredictions\predictions_split_$splitNum.pkl"
    if (Test-Path $predPath) {
      Write-Host ("[{0}] Skip split {1} (exists): {2}" -f (Get-Date -Format "HH:mm:ss"), $splitNum, $predPath)
      continue
    }

    $partLog = Join-Path $logDir ("infer_split_{0}.log" -f $splitNum)
    Write-Host ("[{0}] Running split {1} -> {2}" -f (Get-Date -Format "HH:mm:ss"), $splitNum, $partLog)
    Push-Location $repo
    try {
      # Use conf-thres=0.2 to avoid writing huge AOT pickle files full of detections that will be
      # discarded anyway by the official evaluation threshold (README recommends 0.2).
      $valArgs = @(
        ".\\val.py",
        "--task", "test",
        "--data", $yf.FullName,
        "--weights", $weights,
        "--img", $Img,
        "--batch-size", $BatchSize,
        "--half",
        "--num-frames", $NumFrames,
        "--conf-thres", $ConfThres,
        "--save-aot-predictions"
      )
      if ($ExtraValArgs -and $ExtraValArgs.Count -gt 0) {
        # IMPORTANT: flatten extra args into the native argv list (do not pass as a nested array).
        $valArgs += $ExtraValArgs
      }
      $valArgs += @(
        "--project", $project,
        "--name", $runName,
        "--exist-ok"
      )
      Invoke-Native $py $valArgs $partLog
    } finally {
      Pop-Location
    }
  }
}

if (-not $SkipEval) {
  $predFolder = Join-Path $project "$runName\aotpredictions"
  $evalLog = Join-Path $logDir "eval.log"
  Write-Host ("[{0}] Step 3/3: official airborne metrics -> {1}" -f (Get-Date -Format "HH:mm:ss"), $evalLog)
  Push-Location $repo
  try {
    Invoke-Native $py @(
      ".\\evaluate_aot.py",
      "--results_folder", $predFolder,
      "--evaluation_folder", $evalOut,
      "--detection_threshold", 0.2,
      "--dataset-path", $AOTRoot
    ) $evalLog
  } finally {
    Pop-Location
  }
}

Write-Host ("[{0}] Pipeline done." -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

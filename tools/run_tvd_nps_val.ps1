param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$RepoDir = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone",
  [string]$DataYaml = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\data\NPS_URAP_D.yaml",
  [string]$Weights = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt",
  [string]$Project = "",
  [string]$RunName = "nps_val",
  [int]$BatchSize = 2,
  [int]$Img = 1280,
  [int]$NumFrames = 5,
  [double]$ConfThres = 0.001,
  [double]$IouThres = 0.6,
  [string[]]$ExtraValArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Ensure-Dir([string]$d) {
  New-Item -ItemType Directory -Force -Path $d | Out-Null
}

function Invoke-Native([string]$Exe, [string[]]$ArgList, [string]$LogPath) {
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

$py = Join-Path $RepoDir ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { throw "Missing python venv: $py" }
if (!(Test-Path $DataYaml)) { throw "Missing data yaml: $DataYaml" }
if (!(Test-Path $Weights)) { throw "Missing weights: $Weights" }

if (-not $Project) {
  $Project = Join-Path $RepoDir "runs\val\NPS_URAP"
}

$logDir = Join-Path $URAPRoot "artifacts\logs\nps_val"
Ensure-Dir $logDir
$logPath = Join-Path $logDir ("{0}.log" -f $RunName)

Write-Host ("[{0}] NPS val starting" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host ("Repo: {0}" -f $RepoDir)
Write-Host ("Data: {0}" -f $DataYaml)
Write-Host ("Weights: {0}" -f $Weights)
Write-Host ("Project: {0}" -f $Project)
Write-Host ("RunName: {0}" -f $RunName)
Write-Host ("Log: {0}" -f $logPath)

Push-Location $RepoDir
try {
  $args = @(
    ".\\val.py",
    "--task", "val",
    "--data", $DataYaml,
    "--weights", $Weights,
    "--img", $Img,
    "--batch-size", $BatchSize,
    "--half",
    "--num-frames", $NumFrames,
    "--conf-thres", $ConfThres,
    "--iou-thres", $IouThres
  )
  if ($ExtraValArgs -and $ExtraValArgs.Count -gt 0) {
    # IMPORTANT: flatten extra args into the native argv list (do not pass as a nested array).
    $args += $ExtraValArgs
  }
  $args += @(
    "--project", $Project,
    "--name", $RunName,
    "--exist-ok"
  )
  Invoke-Native $py $args $logPath
} finally {
  Pop-Location
}

$saveDir = Join-Path $Project $RunName
$resultsPath = Join-Path $saveDir "results.txt"
if (Test-Path $resultsPath) {
  Write-Host ("results_txt={0}" -f $resultsPath)
  Get-Content $resultsPath | Select-Object -First 50
} else {
  Write-Host ("WARNING: missing results.txt at {0}" -f $resultsPath)
}

Write-Host ("[{0}] NPS val done." -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

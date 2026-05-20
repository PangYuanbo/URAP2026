param(
  # Put the dataset on a local disk by default (avoid slow/virtual drives like Google Drive).
  [string]$Root = "C:\URAP_datasets\VisDrone",
  [string]$RepoDir = "C:\Users\aaron\Desktop\URAP\papers\ESOD"
)

$ErrorActionPreference = "Stop"

$null = New-Item -ItemType Directory -Force -Path $Root
if (-not (Test-Path -Path $Root -PathType Container)) {
  throw "Failed to create Root directory: $Root"
}

$zipsDir = Join-Path $Root "zips"
New-Item -ItemType Directory -Force -Path $zipsDir | Out-Null

# NOTE: These are mirrors commonly used by the YOLO/Ultralytics ecosystem.
# The dataset license/terms are still governed by VisDrone.
$files = @(
  @{ Name = "VisDrone2019-DET-train.zip"; Url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip" },
  @{ Name = "VisDrone2019-DET-val.zip"; Url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip" },
  @{ Name = "VisDrone2019-DET-test-dev.zip"; Url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip" },
  @{ Name = "VisDrone2019-DET-test-challenge.zip"; Url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-challenge.zip" }
)

foreach ($f in $files) {
  $zipPath = Join-Path $zipsDir $f.Name
  if (-not (Test-Path -Path $zipPath -PathType Leaf)) {
    Write-Host "Downloading $($f.Name) ..."
    curl.exe -L --retry 3 --retry-delay 5 $f.Url -o $zipPath
    if ($LASTEXITCODE -ne 0) {
      throw "curl failed with exit code $LASTEXITCODE for $($f.Name)"
    }
  } else {
    Write-Host "Exists: $($f.Name)"
  }

  $outDirName = [IO.Path]::GetFileNameWithoutExtension($f.Name)
  $outDir = Join-Path $Root $outDirName
  if (-not (Test-Path -Path $outDir -PathType Container)) {
    Write-Host "Extracting $($f.Name) ..."
    Expand-Archive -Path $zipPath -DestinationPath $Root
  } else {
    Write-Host "Extracted: $outDirName"
  }
}

# Make repo-local junction "VisDrone" -> dataset root (Windows-friendly replacement for ln -sf).
$junction = Join-Path $RepoDir "VisDrone"
if (-not (Test-Path -Path $junction)) {
  New-Item -ItemType Junction -Path $junction -Target $Root | Out-Null
  Write-Host "Created junction: $junction -> $Root"
} else {
  Write-Host "Junction already exists: $junction"
}

Write-Host "Done."

param(
  [string]$DatasetDir = "C:\Users\aaron\Desktop\URAP\papers\ESOD\VisDrone",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\ESOD\runs\visdrone_prepare",
  [string]$RunId = "visdrone_prepare",
  [int]$TailLines = 5
)

$ErrorActionPreference = "Stop"

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

$pidValue = $null
if (Test-Path -Path $pidFile -PathType Leaf) {
  $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$proc = $null
if ($pidValue -match '^\d+$') {
  $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
}

Write-Host ("runner_pid={0} running={1}" -f $pidValue, ($null -ne $proc))

$stderrPath = $null
if (Test-Path -Path $metaFile -PathType Leaf) {
  $meta = Get-Content $metaFile
  $stderrLine = $meta | Where-Object { $_ -like "stderr=*" } | Select-Object -First 1
  if ($null -ne $stderrLine) { $stderrPath = $stderrLine.Substring(7) }
}
if ($null -eq $stderrPath) {
  $logsDir = Join-Path $OutputRoot "logs"
  if (Test-Path -Path $logsDir -PathType Container) {
    $stderrPath = (Get-ChildItem -Path $logsDir -Filter "*.err.txt" -File -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName })
  }
}

if ($null -ne $stderrPath -and (Test-Path -Path $stderrPath -PathType Leaf)) {
  $lw = (Get-Item $stderrPath).LastWriteTime
  Write-Host ("stderr={0}" -f $stderrPath)
  Write-Host ("stderr_last_write={0}" -f $lw.ToString("yyyy-MM-dd HH:mm:ss"))
  Write-Host ("--- stderr tail ({0}) ---" -f $TailLines)
  Get-Content -Path $stderrPath -Tail $TailLines -ErrorAction SilentlyContinue
}

if (-not (Test-Path -Path $DatasetDir -PathType Container)) {
  Write-Host ("DatasetDir not found: {0}" -f $DatasetDir)
  exit 0
}

Write-Host ""
Write-Host ("DatasetDir={0}" -f $DatasetDir)

$subsets = @("train", "val", "test-dev")
foreach ($subset in $subsets) {
  $subDir = Join-Path $DatasetDir ("VisDrone2019-DET-{0}" -f $subset)
  if (-not (Test-Path -Path $subDir -PathType Container)) {
    Write-Host ("subset={0} missing_dir={1}" -f $subset, $subDir)
    continue
  }

  $imagesDir = Join-Path $subDir "images"
  $labelsDir = Join-Path $subDir "labels"
  $masksDir = Join-Path $subDir "masks"

  $baseImages = 0
  if (Test-Path -Path $imagesDir -PathType Container) {
    $baseImages = (Get-ChildItem -Path $imagesDir -Filter "*.jpg" -File -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -notlike "*_masked.jpg" } | Measure-Object).Count
  }

  $labelsTxt = 0
  $labelsNpy = 0
  if (Test-Path -Path $labelsDir -PathType Container) {
    $labelsTxt = (Get-ChildItem -Path $labelsDir -Filter "*.txt" -File -ErrorAction SilentlyContinue | Measure-Object).Count
    $labelsNpy = (Get-ChildItem -Path $labelsDir -Filter "*.npy" -File -ErrorAction SilentlyContinue | Measure-Object).Count
  }

  $masksNpy = 0
  if (Test-Path -Path $masksDir -PathType Container) {
    $masksNpy = (Get-ChildItem -Path $masksDir -Filter "*.npy" -File -ErrorAction SilentlyContinue | Measure-Object).Count
  }

  Write-Host ("subset={0} base_images={1} labels_txt={2} labels_npy_legacy={3} masks_npy={4}" -f $subset, $baseImages, $labelsTxt, $labelsNpy, $masksNpy)
}

$splitDir = Join-Path $DatasetDir "split"
if (Test-Path -Path $splitDir -PathType Container) {
  Write-Host ""
  Write-Host ("split_dir={0}" -f $splitDir)
  foreach ($p in (Get-ChildItem -Path $splitDir -Filter "*.txt" -File | Sort-Object Name)) {
    $lines = (Get-Content -Path $p.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    Write-Host ("split_file={0} lines={1}" -f $p.Name, $lines)
  }
}


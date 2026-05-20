param(
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\ESOD\runs\train_visdrone_yolov5m_detached",
  [string]$RunId = "visdrone_yolov5m_e50",
  [int]$TailLines = 30
)

$ErrorActionPreference = "Stop"

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (-not (Test-Path -Path $metaFile -PathType Leaf)) {
  Write-Host "Meta file not found: $metaFile"
  exit 1
}

$meta = Get-Content $metaFile
$pidValue = $null
if (Test-Path -Path $pidFile -PathType Leaf) {
  $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$proc = $null
if ($pidValue -match '^\d+$') {
  $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
}

Write-Host "== Meta =="
$meta | Select-Object -First 120

Write-Host ""
Write-Host ("== Process ==`nRUNNING={0} pid={1}" -f ($null -ne $proc), $pidValue)

$stderrPath = ($meta | Where-Object { $_ -like "stderr=*" } | Select-Object -First 1)
$stderrPath = if ($null -ne $stderrPath) { $stderrPath.Substring(7) } else { $null }

$saveDirLine = ($meta | Where-Object { $_ -like "save_dir=*" } | Select-Object -First 1)
$saveDir = if ($null -ne $saveDirLine) { $saveDirLine.Substring(9) } else { $null }

if ($null -ne $saveDir -and (Test-Path -Path $saveDir -PathType Container)) {
  $resultsFile = Join-Path $saveDir "results.txt"
  $weightsDir = Join-Path $saveDir "weights"
  $lastPt = Join-Path $weightsDir "last.pt"
  $bestPt = Join-Path $weightsDir "best.pt"

  Write-Host ""
  Write-Host ("== SaveDir ==`n{0}" -f $saveDir)

  if (Test-Path -Path $resultsFile -PathType Leaf) {
    $lines = Get-Content $resultsFile -ErrorAction SilentlyContinue
    $n = ($lines | Measure-Object -Line).Lines
    $last = $lines | Select-Object -Last 1
    Write-Host ("results.txt lines={0}" -f $n)
    if ($null -ne $last) { Write-Host ("results.txt last={0}" -f $last) }
  } else {
    Write-Host "results.txt not found yet."
  }

  foreach ($pt in @($lastPt, $bestPt)) {
    if (Test-Path -Path $pt -PathType Leaf) {
      $fi = Get-Item $pt
      Write-Host ("{0} size_mb={1:N1} last_write={2}" -f $fi.Name, ($fi.Length / 1MB), $fi.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"))
    }
  }
}

if ($null -ne $stderrPath -and (Test-Path -Path $stderrPath -PathType Leaf)) {
  $lw = (Get-Item $stderrPath).LastWriteTime
  Write-Host ""
  Write-Host "== Log Tail (stderr) =="
  Write-Host ("stderr={0}" -f $stderrPath)
  Write-Host ("stderr_last_write={0}" -f $lw.ToString("yyyy-MM-dd HH:mm:ss"))
  Get-Content -Path $stderrPath -Tail $TailLines -ErrorAction SilentlyContinue
} else {
  Write-Host ""
  Write-Host "stderr log not found yet."
}


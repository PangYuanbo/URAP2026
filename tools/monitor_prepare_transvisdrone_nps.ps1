param(
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_sota_research\prepare_transvisdrone_nps_runner",
  [string]$RunId = "prepare_transvisdrone_nps",
  [int]$TailLines = 40
)

$ErrorActionPreference = "Stop"
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

Write-Host "== Meta =="
if (Test-Path $metaFile) { Get-Content $metaFile } else { Write-Host "meta missing: $metaFile" }

$meta = @{}
if (Test-Path $metaFile) {
  foreach ($line in Get-Content $metaFile) {
    $idx = $line.IndexOf("=")
    if ($idx -gt 0) { $meta[$line.Substring(0, $idx)] = $line.Substring($idx + 1) }
  }
}

$pidValue = $null
if (Test-Path $pidFile) { $pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1 }
$process = $null
if ($pidValue) { $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue }
if ($process -and $process.CommandLine -like "*prepare_transvisdrone_nps.py*") {
  Write-Host ""
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else {
  Write-Host ""
  Write-Host "NOT RUNNING PID=$pidValue"
}

$outRoot = $meta["out_root"]
$split = $meta["only_split"]
$stdout = $meta["stdout"]
$stderr = $meta["stderr"]
$framesDir = if ($outRoot -and $split) { Join-Path $outRoot "AllFrames\$split" } else { $null }
$labelsDir = if ($outRoot -and $split) { Join-Path $outRoot "NPSvisdroneStyle\$split\labels" } else { $null }
$videoLen = if ($outRoot -and $split) { Join-Path $outRoot "Videos\$split\video_length_dict.pkl" } else { $null }

$frameCount = if ($framesDir -and (Test-Path $framesDir)) { (Get-ChildItem -LiteralPath $framesDir -Filter "*.png" -ErrorAction SilentlyContinue | Measure-Object).Count } else { 0 }
$labelCount = if ($labelsDir -and (Test-Path $labelsDir)) { (Get-ChildItem -LiteralPath $labelsDir -Filter "*.txt" -ErrorAction SilentlyContinue | Measure-Object).Count } else { 0 }

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $videoLen)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}
if ($framesDir -and (Test-Path $framesDir)) {
  $latestFrame = Get-ChildItem -LiteralPath $framesDir -Filter "*.png" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
  if ($latestFrame -and (-not $lastWrite -or $latestFrame.LastWriteTime -gt $lastWrite)) { $lastWrite = $latestFrame.LastWriteTime }
}

$done = if ($videoLen -and (Test-Path $videoLen)) { 1 } else { 0 }
Write-Host ""
Write-Host "done/total: $done/1"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: frames=$frameCount labels=$labelCount video_length_dict=$videoLen"
Write-Host "frames dir: $framesDir"
Write-Host "labels dir: $labelsDir"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }

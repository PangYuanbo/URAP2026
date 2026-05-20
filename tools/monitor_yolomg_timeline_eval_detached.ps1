param(
  [string]$OutputRoot = 'C:/Users/aaron/Desktop/URAP/URAP-UAV-to-UAV-Detection-and-Tracking/papers/YOLOMG/runs/detached_timeline_eval',
  [string]$RunId = 'yolomg_timeline_eval',
  [int]$TailLines = 80
)

$ErrorActionPreference = 'Stop'

$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (-not (Test-Path -Path $metaFile -PathType Leaf)) {
  Write-Host "Meta file not found: $metaFile"
  exit 1
}

$meta = Get-Content $metaFile
Write-Host '== Meta =='
$meta | Select-Object -First 120

$pidValue = $null
if (Test-Path -Path $pidFile -PathType Leaf) {
  $pidValue = (Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
}

$proc = $null
if ($pidValue -match '^\d+$') {
  $proc = Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue
}

Write-Host ''
if ($null -ne $proc) {
  Write-Host ('RUNNING=true PID={0}' -f $pidValue)
  Write-Host ('PID_START={0}' -f $proc.StartTime.ToString('yyyy-MM-dd HH:mm:ss'))
} else {
  Write-Host ('NOT RUNNING PID={0}' -f $pidValue)
}

$stdoutLine = ($meta | Where-Object { $_ -like 'stdout=*' } | Select-Object -First 1)
$stderrLine = ($meta | Where-Object { $_ -like 'stderr=*' } | Select-Object -First 1)
$outputDirLine = ($meta | Where-Object { $_ -like 'output_dir=*' } | Select-Object -First 1)
$stdoutPath = if ($stdoutLine) { $stdoutLine.Substring(7) } else { $null }
$stderrPath = if ($stderrLine) { $stderrLine.Substring(7) } else { $null }
$outputDir = if ($outputDirLine) { $outputDirLine.Substring(11) } else { $null }

if ($outputDir -and (Test-Path -Path $outputDir -PathType Container)) {
  Write-Host ('OUTPUT_DIR={0}' -f $outputDir)
  $manifest = Join-Path $outputDir 'manifest.json'
  if (Test-Path -Path $manifest -PathType Leaf) {
    $manifestItem = Get-Item $manifest
    Write-Host ('MANIFEST_LAST_WRITE={0}' -f $manifestItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  }

  $perFrameDir = Join-Path $outputDir 'per_frame'
  if (Test-Path -Path $perFrameDir -PathType Container) {
    $csvCount = (Get-ChildItem -Path $perFrameDir -Filter '*.csv' -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host ('PER_FRAME_CSVS={0}' -f $csvCount)
  }

  $plotsDir = Join-Path $outputDir 'plots'
  if (Test-Path -Path $plotsDir -PathType Container) {
    $plotCount = (Get-ChildItem -Path $plotsDir -Filter '*.png' -File -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Host ('PLOTS={0}' -f $plotCount)
  }
}

if ($stdoutPath -and (Test-Path -Path $stdoutPath -PathType Leaf)) {
  $stdoutItem = Get-Item $stdoutPath
  Write-Host ('STDOUT_LOG={0}' -f $stdoutPath)
  Write-Host ('STDOUT_LAST_WRITE={0}' -f $stdoutItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  $stdoutTail = Get-Content -Path $stdoutPath -Tail 1200 -ErrorAction SilentlyContinue
  $stdoutText = $stdoutTail -join "`n"
  $progressMatches = [regex]::Matches($stdoutText, 'timeline-eval:\s+(\d+)%\|.*?\|\s*(\d+)/(\d+)')
  if ($progressMatches.Count -gt 0) {
    $m = $progressMatches[$progressMatches.Count - 1]
    Write-Host ('PROGRESS_DONE={0}' -f $m.Groups[2].Value)
    Write-Host ('PROGRESS_TOTAL={0}' -f $m.Groups[3].Value)
  } elseif ($outputDir -and (Test-Path -Path (Join-Path $outputDir 'manifest.json') -PathType Leaf)) {
    $manifestJson = Get-Content -Path (Join-Path $outputDir 'manifest.json') -Raw | ConvertFrom-Json
    Write-Host ('PROGRESS_DONE={0}' -f $manifestJson.total_frames)
    Write-Host ('PROGRESS_TOTAL={0}' -f $manifestJson.total_frames)
  } else {
    Write-Host 'PROGRESS_DONE=NA'
    Write-Host 'PROGRESS_TOTAL=NA'
  }
  Write-Host ''
  Write-Host '== stdout tail =='
  $stdoutTail | Select-Object -Last $TailLines
}

if ($stderrPath -and (Test-Path -Path $stderrPath -PathType Leaf)) {
  $stderrItem = Get-Item $stderrPath
  $stderrTail = Get-Content -Path $stderrPath -Tail 1200 -ErrorAction SilentlyContinue
  $stderrText = $stderrTail -join "`n"
  if (($stdoutPath -and -not $progressMatches) -or (-not $stdoutPath)) {
    $progressMatches = [regex]::Matches($stderrText, '(\d+)/(\d+)')
    if ($progressMatches.Count -gt 0) {
      $m = $progressMatches[$progressMatches.Count - 1]
      Write-Host ('PROGRESS_DONE={0}' -f $m.Groups[1].Value)
      Write-Host ('PROGRESS_TOTAL={0}' -f $m.Groups[2].Value)
    }
  }
  Write-Host ''
  Write-Host ('STDERR_LOG={0}' -f $stderrPath)
  Write-Host ('STDERR_LAST_WRITE={0}' -f $stderrItem.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  Write-Host '== stderr tail =='
  $stderrTail | Select-Object -Last $TailLines
}

$gpuLine = (& nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1)
if ($LASTEXITCODE -eq 0 -and $gpuLine) {
  Write-Host ''
  Write-Host ('GPU_SIGNAL={0}' -f $gpuLine)
}

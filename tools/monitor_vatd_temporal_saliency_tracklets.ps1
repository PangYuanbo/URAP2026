param(
  [string]$RunId = 'vatd_temporal_saliency_tracklets',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\vatd_temporal_saliency_tracklets_runner'),
  [int]$TailLines = 40
)

$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

Write-Host '== Meta =='
if (Test-Path $metaFile) {
  Get-Content $metaFile
} else {
  Write-Host "meta missing: $metaFile"
}

$pidValue = $null
if (Test-Path $pidFile) {
  $pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
}

$process = $null
if ($pidValue) {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
}

if ($process -and $process.CommandLine -like '*export-temporal-saliency-tracklets*') {
  Write-Host ""
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id $pidValue -ErrorAction SilentlyContinue).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else {
  Write-Host ""
  Write-Host "NOT RUNNING PID=$pidValue"
}

$meta = @{}
if (Test-Path $metaFile) {
  foreach ($line in Get-Content $metaFile) {
    $idx = $line.IndexOf('=')
    if ($idx -gt 0) { $meta[$line.Substring(0, $idx)] = $line.Substring($idx + 1) }
  }
}

$stdout = $meta['stdout']
$stderr = $meta['stderr']
$outDir = $meta['out_dir']
$jsonl = if ($outDir) { Join-Path $outDir 'proposal_tracklets.jsonl' } else { $null }
$csv = if ($outDir) { Join-Path $outDir 'proposal_tracklets.csv' } else { $null }
$summary = if ($outDir) { Join-Path $outDir 'summary.json' } else { $null }

$lastProgress = $null
if ($stdout -and (Test-Path $stdout)) {
  $progressLines = Get-Content $stdout | Where-Object { $_ -like '*export_temporal_saliency_tracklets_progress*' }
  if ($progressLines) { $lastProgress = $progressLines | Select-Object -Last 1 }
}

$done = 0
$total = 0
$lastUnit = ''
if ($lastProgress) {
  try {
    $progress = $lastProgress | ConvertFrom-Json
    $done = [int]$progress.sequences_done
    $total = [int]$progress.sequences_total
    $lastUnit = "seq=$($progress.last_seq) tracklets=$($progress.tracklets) positives=$($progress.positives) candidates=$($progress.candidate_rows)"
  } catch {
    $lastUnit = 'progress_parse_failed'
  }
}

$jsonlLines = 0
if ($jsonl -and (Test-Path $jsonl)) {
  $jsonlLines = (Get-Content $jsonl | Measure-Object -Line).Lines
}

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $jsonl, $csv, $summary)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

Write-Host ""
Write-Host "done/total: $done/$total"
Write-Host "jsonl lines: $jsonlLines"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: $lastUnit"
Write-Host "jsonl: $jsonl"
Write-Host "csv: $csv"
Write-Host "summary: $summary"
Write-Host "stdout log: $stdout"
Write-Host "stderr log: $stderr"

Write-Host ""
Write-Host "== GPU signal =="
$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
  & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
} else {
  Write-Host "nvidia-smi not found"
}

Write-Host ""
Write-Host "== stdout tail =="
if ($stdout -and (Test-Path $stdout)) { Get-Content $stdout -Tail $TailLines }
Write-Host ""
Write-Host "== stderr tail =="
if ($stderr -and (Test-Path $stderr)) { Get-Content $stderr -Tail $TailLines }

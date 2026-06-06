param(
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_sota_research\tvd_nps_eval_runner",
  [string]$RunId = "tvd_nps_eval",
  [int]$TailLines = 40
)

$ErrorActionPreference = "Stop"
$pidFile = Join-Path $OutputRoot "$RunId.pid"
$metaFile = Join-Path $OutputRoot "$RunId.meta.txt"

Write-Host "== Meta =="
if (Test-Path $metaFile) {
  Get-Content $metaFile
} else {
  Write-Host "meta missing: $metaFile"
}

$meta = @{}
if (Test-Path $metaFile) {
  foreach ($line in Get-Content $metaFile) {
    $idx = $line.IndexOf("=")
    if ($idx -gt 0) { $meta[$line.Substring(0, $idx)] = $line.Substring($idx + 1) }
  }
}

$pidValue = $null
if (Test-Path $pidFile) {
  $pidValue = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
}
$process = $null
if ($pidValue) {
  $process = Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue
}
if ($process -and $process.CommandLine -like "*val.py*") {
  Write-Host ""
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue) -ErrorAction SilentlyContinue).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else {
  Write-Host ""
  Write-Host "NOT RUNNING PID=$pidValue"
}

$stdout = $meta["stdout"]
$stderr = $meta["stderr"]
$project = $meta["project"]
$runName = $meta["run_name"]
$runDir = if ($project -and $runName) { Join-Path $project $runName } else { $null }
$results = if ($runDir) { Join-Path $runDir "results.txt" } else { $null }
$pred = if ($runDir) { Join-Path $runDir "predictionsgt\predictionsgt_split_0.pkl" } else { $null }

$lastWrite = $null
foreach ($path in @($stdout, $stderr, $results, $pred)) {
  if ($path -and (Test-Path $path)) {
    $t = (Get-Item $path).LastWriteTime
    if (-not $lastWrite -or $t -gt $lastWrite) { $lastWrite = $t }
  }
}

$done = 0
$total = 1
$lastUnit = ""
if ($pred -and (Test-Path $pred)) {
  $done = 1
  $lastUnit = "predictionsgt_split_0.pkl"
} elseif ($stdout -and (Test-Path $stdout)) {
  $lastLine = Get-Content $stdout -Tail 200 | Where-Object { $_ -match "^\s*\d+/" -or $_ -like "*all*" } | Select-Object -Last 1
  if ($lastLine) { $lastUnit = $lastLine }
}

Write-Host ""
Write-Host "done/total: $done/$total"
Write-Host "last output timestamp: $lastWrite"
Write-Host "last completed unit: $lastUnit"
Write-Host "run dir: $runDir"
Write-Host "results: $results"
Write-Host "predictionsgt: $pred"
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

param(
  [string]$RunnerRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\train_runner",
  [string]$RunId = "nps_yolomg_train50",
  [int]$TailLines = 30
)

$ErrorActionPreference = "Stop"
$pidFile = Join-Path $RunnerRoot "$RunId.pid"
$metaFile = Join-Path $RunnerRoot "$RunId.meta.txt"
$meta = @{}
if (Test-Path $metaFile) { foreach ($line in Get-Content $metaFile) { $index = $line.IndexOf('='); if ($index -gt 0) { $meta[$line.Substring(0, $index)] = $line.Substring($index + 1) } } }
$pidValue = if (Test-Path $pidFile) { Get-Content $pidFile | Select-Object -First 1 } else { $null }
$process = if ($pidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
if ($process -and $process.CommandLine -like "*train.py*") {
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue)).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
  $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $pidValue" -ErrorAction SilentlyContinue
  foreach ($child in $children) { Write-Host "CHILD_PID=$($child.ProcessId) CHILD_COMMAND=$($child.CommandLine)" }
} else { Write-Host "NOT RUNNING PID=$pidValue" }
$resultsCsv = if ($meta['run_dir']) { Join-Path $meta['run_dir'] 'results.csv' } else { $null }
$resultsTxt = if ($meta['run_dir']) { Join-Path $meta['run_dir'] 'results.txt' } else { $null }
$done = 0
if ($resultsCsv -and (Test-Path $resultsCsv)) { $done = [Math]::Max(0, (Get-Content $resultsCsv | Measure-Object -Line).Lines - 1) }
elseif ($resultsTxt -and (Test-Path $resultsTxt)) { $done = (Get-Content $resultsTxt | Where-Object { $_ -match '^\s*\d+/49' } | Measure-Object).Count }
Write-Host "done/total: $done/50"
$lastCheckpoint = if ($meta['run_dir']) { Join-Path $meta['run_dir'] 'weights\last.pt' } else { $null }
$latest = @($resultsCsv, $resultsTxt, $lastCheckpoint, $meta['stdout'], $meta['stderr']) | Where-Object { $_ -and (Test-Path $_) } | ForEach-Object { Get-Item $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "last output timestamp: $(if($latest){$latest.LastWriteTime}else{'none'})"
Write-Host "last completed unit: epoch=$done checkpoint=$lastCheckpoint"
Write-Host "stdout log: $($meta['stdout'])"
Write-Host "stderr log: $($meta['stderr'])"
Write-Host "== GPU signal =="
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits } else { Write-Host "nvidia-smi unavailable" }
if ($meta['stdout'] -and (Test-Path $meta['stdout'])) { Write-Host "== stdout tail =="; Get-Content $meta['stdout'] -Tail $TailLines }
if ($meta['stderr'] -and (Test-Path $meta['stderr'])) { Write-Host "== stderr tail =="; Get-Content $meta['stderr'] -Tail $TailLines }

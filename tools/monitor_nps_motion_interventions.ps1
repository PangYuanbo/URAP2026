param(
  [string]$RunnerRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\dataset_builder",
  [string]$RunId = "nps_motion_interventions",
  [int]$TailLines = 30
)

$ErrorActionPreference = "Stop"
$pidFile = Join-Path $RunnerRoot "$RunId.pid"
$metaFile = Join-Path $RunnerRoot "$RunId.meta.txt"
$meta = @{}
if (Test-Path $metaFile) { foreach ($line in Get-Content $metaFile) { $index = $line.IndexOf('='); if ($index -gt 0) { $meta[$line.Substring(0, $index)] = $line.Substring($index + 1) } } }
$pidValue = if (Test-Path $pidFile) { Get-Content $pidFile | Select-Object -First 1 } else { $null }
$process = if ($pidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
if ($process -and $process.CommandLine -like "*build_nps_motion_interventions.py*") {
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue)).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else { Write-Host "NOT RUNNING PID=$pidValue" }
$progressPath = if ($meta['out_root']) { Join-Path $meta['out_root'] 'progress.json' } else { $null }
$progress = if ($progressPath -and (Test-Path $progressPath)) { Get-Content $progressPath -Raw | ConvertFrom-Json } else { $null }
Write-Host "done/total: $(if($progress){$progress.done}else{0})/$(if($progress){$progress.total}else{'?'})"
Write-Host "last completed unit: $(if($progress){$progress.last_completed_unit}else{'none'})"
$latest = @($progressPath, $meta['stdout'], $meta['stderr']) | Where-Object { $_ -and (Test-Path $_) } | ForEach-Object { Get-Item $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "last output timestamp: $(if($latest){$latest.LastWriteTime}else{'none'})"
Write-Host "progress path: $progressPath"
Write-Host "stdout log: $($meta['stdout'])"
Write-Host "stderr log: $($meta['stderr'])"
if ($meta['stdout'] -and (Test-Path $meta['stdout'])) { Write-Host "== stdout tail =="; Get-Content $meta['stdout'] -Tail $TailLines }
if ($meta['stderr'] -and (Test-Path $meta['stderr'])) { Write-Host "== stderr tail =="; Get-Content $meta['stderr'] -Tail $TailLines }

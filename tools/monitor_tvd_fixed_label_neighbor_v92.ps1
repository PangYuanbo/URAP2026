$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repo 'artifacts\detached_tvd_fixed_label_neighbor_v92'
$pidFile = Join-Path $runDir 'pid.txt'
$progressFile = Join-Path $runDir 'progress.json'
$metaFile = Join-Path $runDir 'start_meta.json'
$stdout = Join-Path $runDir 'stdout.log'
$stderr = Join-Path $runDir 'stderr.log'
if (!(Test-Path -LiteralPath $pidFile)) { Write-Output 'NOT RUNNING: no PID file'; exit 1 }
$jobPid = [int](Get-Content -LiteralPath $pidFile -Raw).Trim()
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$jobPid" -ErrorAction SilentlyContinue
$meta = if (Test-Path -LiteralPath $metaFile) { Get-Content -LiteralPath $metaFile -Raw | ConvertFrom-Json } else { $null }
$progress = if (Test-Path -LiteralPath $progressFile) { Get-Content -LiteralPath $progressFile -Raw | ConvertFrom-Json } else { $null }
if ($process) { Write-Output "RUNNING: PID=$jobPid START=$($meta.start_time)"; Write-Output "COMMAND=$($process.CommandLine)" } else { Write-Output "NOT RUNNING: PID=$jobPid START=$($meta.start_time)" }
if ($progress) {
    Write-Output "PROGRESS=$($progress.done)/$($progress.total) STAGE=$($progress.stage) UPDATED=$($progress.updated) CHILD_PID=$($progress.child_pid)"
    if ($progress.child_pid) { $child = Get-CimInstance Win32_Process -Filter "ProcessId=$($progress.child_pid)" -ErrorAction SilentlyContinue; Write-Output "CHILD_ALIVE=$([bool]$child)" }
}
foreach ($log in @($stdout,$stderr)) { if (Test-Path -LiteralPath $log) { $item=Get-Item -LiteralPath $log; Write-Output "LOG=$log LAST_WRITE=$($item.LastWriteTime.ToString('o')) BYTES=$($item.Length)"; Get-Content -LiteralPath $log -Tail 10 } }
$nvidiaSmi=Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaSmi) { & nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits }

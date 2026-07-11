$ErrorActionPreference = 'Stop'
$runDir = 'C:\Users\aaron\Desktop\URAP\artifacts\detached_ard100_yolomg_prepare_v2'
$meta = Get-Content (Join-Path $runDir 'meta.json') -Raw | ConvertFrom-Json
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
$progressPath = Join-Path $runDir 'progress.json'
$progress = if (Test-Path $progressPath) { Get-Content $progressPath -Raw | ConvertFrom-Json } else { $null }
$lastOutput = if (Test-Path $meta.stderr_log) { (Get-Item $meta.stderr_log).LastWriteTime.ToString('o') } elseif (Test-Path $meta.stdout_log) { (Get-Item $meta.stdout_log).LastWriteTime.ToString('o') } else { $null }
[pscustomobject]@{
    status = if ($process) { 'RUNNING' } else { 'NOT RUNNING' }
    done = if ($progress) { [int]$progress.done } else { 0 }
    total = if ($progress) { [int]$progress.total } else { 3 }
    pid = [int]$meta.pid
    command_line = if ($process) { $process.CommandLine } else { $null }
    start_time = $meta.start_time
    last_output_timestamp = $lastOutput
    last_completed_unit = if ($progress) { $progress.stage } else { 'not_started' }
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
} | ConvertTo-Json -Depth 3

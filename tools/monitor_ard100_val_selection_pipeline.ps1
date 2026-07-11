$ErrorActionPreference = 'Stop'
$runDir = 'C:\Users\aaron\Desktop\URAP\artifacts\detached_ard100_val_selection_pipeline_v2'
$meta = Get-Content (Join-Path $runDir 'meta.json') -Raw | ConvertFrom-Json
$progressPath = Join-Path $runDir 'progress.json'
$progress = if (Test-Path $progressPath) { Get-Content $progressPath -Raw | ConvertFrom-Json } else { $null }
$all = @(Get-CimInstance Win32_Process)
$ids = New-Object System.Collections.Generic.HashSet[int]
[void]$ids.Add([int]$meta.pid)
do {
    $before = $ids.Count
    foreach ($process in $all) { if ($ids.Contains([int]$process.ParentProcessId)) { [void]$ids.Add([int]$process.ProcessId) } }
} while ($ids.Count -gt $before)
$matching = @($all | Where-Object { $ids.Contains([int]$_.ProcessId) })
[pscustomobject]@{
    status = if ($matching.Count) { 'RUNNING' } else { 'NOT RUNNING' }
    done = if ($progress) { [int]$progress.done } else { 0 }
    total = if ($progress) { [int]$progress.total } else { 3 }
    launcher_pid = [int]$meta.pid
    compute_pids = @($matching | Select-Object -ExpandProperty ProcessId -Unique)
    start_time = $meta.start_time
    last_output_timestamp = if (Test-Path $meta.stdout_log) { (Get-Item $meta.stdout_log).LastWriteTime.ToString('o') } else { $null }
    last_completed_unit = if ($progress) { $progress.stage } else { 'not_started' }
    progress = $progress
    stdout_log = $meta.stdout_log
    stderr_log = $meta.stderr_log
} | ConvertTo-Json -Depth 8

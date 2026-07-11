$ErrorActionPreference = 'Stop'
$runDir = 'C:\Users\aaron\Desktop\URAP\artifacts\detached_ard100_yolomg_val_corrected_v2'
$meta = Get-Content (Join-Path $runDir 'meta.json') -Raw | ConvertFrom-Json
$all = @(Get-CimInstance Win32_Process)
$ids = New-Object System.Collections.Generic.HashSet[int]
[void]$ids.Add([int]$meta.pid)
do {
    $before = $ids.Count
    foreach ($process in $all) {
        if ($ids.Contains([int]$process.ParentProcessId)) { [void]$ids.Add([int]$process.ProcessId) }
    }
} while ($ids.Count -gt $before)
$matching = @($all | Where-Object { $ids.Contains([int]$_.ProcessId) })
$text = if (Test-Path $meta.stderr_log) { Get-Content $meta.stderr_log -Raw } else { '' }
$progress = [regex]::Matches($text, '(\d+)/(' + $meta.total_batches + ')')
$done = if ($progress.Count) { [int]$progress[$progress.Count - 1].Groups[1].Value } else { 0 }
$metrics = [regex]::Matches($text, 'all\s+20762\s+\d+\s+[0-9.eE+\-]+\s+[0-9.eE+\-]+\s+[0-9.eE+\-]+\s+[0-9.eE+\-]+')
$final = if ($metrics.Count) { $metrics[$metrics.Count - 1].Value.Trim() } else { $null }
if ($final) { $done = [int]$meta.total_batches }
$pids = @($matching | Select-Object -ExpandProperty ProcessId -Unique)
$gpuRows = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null)
$gpu = @($gpuRows | Where-Object { $row = $_; $pids | Where-Object { $row -match "^$($_)," } })
[pscustomobject]@{
    status = if ($matching.Count) { 'RUNNING' } else { 'NOT RUNNING' }
    done = $done
    total = [int]$meta.total_batches
    launcher_pid = [int]$meta.pid
    compute_pids = $pids
    start_time = $meta.start_time
    last_output_timestamp = if (Test-Path $meta.stderr_log) { (Get-Item $meta.stderr_log).LastWriteTime.ToString('o') } else { $null }
    last_completed_unit = if ($final) { 'evaluation complete' } elseif ($done) { "batch $done/$($meta.total_batches)" } else { 'cache scan or warmup' }
    final_metrics = $final
    stderr_log = $meta.stderr_log
    output_dir = $meta.output_dir
    gpu_process_signal = ($gpu -join '; ')
} | ConvertTo-Json -Depth 3

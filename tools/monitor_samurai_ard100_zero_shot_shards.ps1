param()
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$metas = @(Get-ChildItem $controlRoot -Filter "ard100_ablation_*zero_shot_test_v1_shard*.meta.json")
$rows = foreach ($metaFile in $metas) {
    $meta = Get-Content $metaFile.FullName -Raw | ConvertFrom-Json
    $pidValue = [int]$meta.pid
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
    $matches = $parent -and $parent.Name -eq "python.exe" -and $parent.CommandLine -like "*eval_samurai_nps.py*"
    $children = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object { $_.ParentProcessId -eq $pidValue -and $_.Name -eq "python.exe" -and $_.CommandLine -like "*eval_samurai_nps.py*" })
    $progress = if (Test-Path $meta.progress_file) { Get-Content $meta.progress_file -Raw | ConvertFrom-Json } else { $null }
    $stderr = Get-Item $meta.stderr_log -ErrorAction SilentlyContinue
    $errorTail = if ($stderr -and $stderr.Length -gt 0) { @(Get-Content $stderr.FullName -Tail 100 | Select-String -Pattern "Traceback|CUDA out of memory|RuntimeError:") } else { @() }
    [ordered]@{
        name = $meta.control_name
        status = if ($matches) { "RUNNING" } elseif ($progress -and $progress.status -eq "completed") { "COMPLETE" } elseif ($errorTail.Count) { "NOT RUNNING (ERROR)" } else { "NOT RUNNING" }
        done_total = if ($progress) { "$($progress.done_sequences)/$($progress.total_sequences)" } else { "0/?" }
        done_frames = if ($progress) { [int]$progress.done_frames } else { 0 }
        pid = $pidValue
        compute_pids = @($pidValue) + @($children.ProcessId)
        start_time = $meta.started_at
        last_completed_unit = if ($progress) { $progress.last_completed_sequence } else { $null }
        last_output_timestamp = if ($stderr) { $stderr.LastWriteTime.ToString("o") } else { $null }
        stderr_log = $meta.stderr_log
        progress_file = $meta.progress_file
    }
}
$gpu = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>$null
[ordered]@{ jobs = $rows; gpu = $gpu } | ConvertTo-Json -Depth 6

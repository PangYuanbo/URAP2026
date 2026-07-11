param()
$ErrorActionPreference = "Stop"
$runDir = "C:\Users\aaron\Desktop\URAP\artifacts\detached_ard100_tvd_candidates_v1"
$metaFile = Join-Path $runDir "meta.json"
if (-not (Test-Path $metaFile)) { throw "No ARD100 run metadata found" }
$meta = Get-Content $metaFile -Raw | ConvertFrom-Json
$allProcesses = Get-CimInstance Win32_Process
$rootProcess = $allProcesses | Where-Object { $_.ProcessId -eq [int]$meta.pid }
$children = @($allProcesses | Where-Object { $_.ParentProcessId -eq [int]$meta.pid })
$matching = @($rootProcess) + $children | Where-Object { $_ -and $_.CommandLine -match "inference.py" -and $_.CommandLine -match "ARD100" }
$alive = $matching.Count -gt 0
$computePids = @($matching | Select-Object -ExpandProperty ProcessId -Unique)
$log = $meta.stderr_log
$done = 0
$lastUnit = "not_started"
if (Test-Path $log) {
  $content = Get-Content $log -Raw -ErrorAction SilentlyContinue
  $matches = [regex]::Matches($content, "(\d+)/($($meta.total_batches))")
  if ($matches.Count -gt 0) { $done = [int]$matches[$matches.Count - 1].Groups[1].Value; $lastUnit = "batch $done/$($meta.total_batches)" }
}
$pkl = Join-Path $meta.output_dir "aotpredictions\predictions_split_0.pkl"
if (Test-Path $pkl) { $done = [int]$meta.total_batches; $lastUnit = "candidate PKL complete" }
$lastOutput = if (Test-Path $pkl) {(Get-Item $pkl).LastWriteTime.ToString("o")} elseif(Test-Path $log){(Get-Item $log).LastWriteTime.ToString("o")} else {$null}
$gpuRows = @(& nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null)
$gpuSignal = @($gpuRows | Where-Object { $row = $_; $computePids | Where-Object { $row -match "^$($_)," } })
[pscustomobject]@{
 status = if($alive){"RUNNING"}else{"NOT RUNNING"}; done=$done; total=[int]$meta.total_batches; launcher_pid=[int]$meta.pid; compute_pids=$computePids;
 start_time=$meta.start_time; last_output_timestamp=$lastOutput; last_completed_unit=$lastUnit;
 stdout_log=$meta.stdout_log; stderr_log=$meta.stderr_log; output_pkl=$pkl; gpu_process_signal=($gpuSignal -join "; ");
 command_lines=@($matching | Select-Object -ExpandProperty CommandLine)
} | ConvertTo-Json -Depth 4

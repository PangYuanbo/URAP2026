param([int]$TailLines = 20)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runRoot = Join-Path $repoRoot "artifacts\joint_training\yolomg_snapshot_watcher"
$pidPath = Join-Path $runRoot "snapshot.pid"
$metaPath = Join-Path $runRoot "snapshot.meta.json"
$meta = if (Test-Path -LiteralPath $metaPath) { Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json } else { $null }
$pidValue = if (Test-Path -LiteralPath $pidPath) { Get-Content -LiteralPath $pidPath | Select-Object -First 1 } else { $null }
$process = if ($pidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
$valid = $process -and $process.CommandLine -like "*yolomg_checkpoint_snapshot_worker.ps1*"
$snapshots = if ($meta) { @(Get-ChildItem -LiteralPath (Join-Path $meta.run_dir "weights\time_snapshots") -Filter "snapshot_*.pt" -File -ErrorAction SilentlyContinue) } else { @() }
$latest = $snapshots | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($valid) { Write-Output "RUNNING"; Write-Output "pid: $pidValue" } else { Write-Output "NOT RUNNING"; Write-Output "last_pid: $pidValue" }
Write-Output "done/total: $($snapshots.Count)/time-based"
Write-Output "start_time: $(if($meta){$meta.start_time}else{'unknown'})"
Write-Output "last_completed_unit: $(if($latest){$latest.Name}else{'none'})"
Write-Output "last_output_timestamp: $(if($latest){$latest.LastWriteTime.ToString('o')}else{'none'})"
Write-Output "interval_seconds: $(if($meta){$meta.interval_seconds}else{'unknown'})"
Write-Output "stdout: $(if($meta){$meta.stdout}else{'none'})"
Write-Output "stderr: $(if($meta){$meta.stderr}else{'none'})"
if ($meta -and (Test-Path -LiteralPath $meta.stdout)) { Get-Content -LiteralPath $meta.stdout -Tail $TailLines }
if ($meta -and (Test-Path -LiteralPath $meta.stderr)) { Get-Content -LiteralPath $meta.stderr -Tail $TailLines }

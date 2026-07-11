$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Run=Join-Path $Repo 'artifacts\detached_nps_samurai_native_clip38_v8'
$StatePath=Join-Path $Run 'state.json'
if (-not (Test-Path $StatePath)) { Write-Host 'status: NOT RUNNING'; Write-Host 'done/total: 0/1800'; exit 0 }
$State=Get-Content $StatePath -Raw | ConvertFrom-Json
$Process=Get-CimInstance Win32_Process -Filter "ProcessId=$($State.pid)" -ErrorAction SilentlyContinue
$Children=@(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $State.pid -and $_.CommandLine -like '*score_predictionsgt_samurai_native.py*' })
$Progress=$null
if (Test-Path $State.progress) { $Progress=Get-Content $State.progress -Raw | ConvertFrom-Json }
Write-Host ("status: " + $(if ($Process) {'RUNNING'} else {'NOT RUNNING'}))
Write-Host ("done/total: " + $(if ($Progress) {"$($Progress.done)/$($Progress.total)"} else {"0/$($State.total)"}))
Write-Host ("stage: " + $(if ($Progress) {$Progress.stage} else {'loading_model'}))
Write-Host "pid: $($State.pid)"
Write-Host "start_time: $($State.start_time)"
if ($Process) { Write-Host "command: $($Process.CommandLine)" }
if ($Children.Count -gt 0) { Write-Host "runtime_pid: $($Children[0].ProcessId)"; Write-Host "runtime_command: $($Children[0].CommandLine)" } else { Write-Host 'runtime_pid: none' }
$Paths=@($State.stdout,$State.stderr,$State.progress) | Where-Object { Test-Path $_ }
$Latest=$Paths | ForEach-Object { Get-Item $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host ("last_output_timestamp: " + $(if ($Latest) {$Latest.LastWriteTime} else {'none'}))
Write-Host ("last_completed_unit: " + $(if ($Progress) {($Progress | ConvertTo-Json -Compress)} else {'none'}))
$GpuPids=@($State.pid)+@($Children | ForEach-Object {$_.ProcessId})
$GpuRows=& nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null
$Gpu=@($GpuRows | Where-Object {$row=$_;$GpuPids | Where-Object {$row -match "^\s*$_,"}})
$DeviceGpu=& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
Write-Host ("gpu_signal: " + $(if ($Gpu) {($Gpu -join '; ') + "; device=" + ($DeviceGpu -join '; ')} elseif ($Children.Count -gt 0) {"device=" + ($DeviceGpu -join '; ')} else {'NO GPU SIGNAL YET'}))
Write-Host "stdout: $($State.stdout)"
Write-Host "stderr: $($State.stderr)"
Write-Host "progress: $($State.progress)"

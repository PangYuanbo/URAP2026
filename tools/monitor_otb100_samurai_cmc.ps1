param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='otb100_samurai_cmc_timebank_v2')
$root=Join-Path $RepoRoot 'artifacts\detached_otb100_samurai_cmc_timebank_v2'
$pidFile=Join-Path $root ($RunId+'.pid');$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0}
$process=if($pidValue){Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue}else{$null}
if($process -and $process.CommandLine -notlike '*evaluate_samurai_otb100.py*'){$process=$null}
$children=if($pidValue){@(Get-CimInstance Win32_Process | Where-Object {$_.ParentProcessId -eq $pidValue -and $_.CommandLine -like '*evaluate_samurai_otb100.py*'})}else{@()}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$progress=if($meta -and (Test-Path $meta.progress)){Get-Content $meta.progress -Raw|ConvertFrom-Json}else{$null}
$paths=if($meta){@($meta.stdout,$meta.stderr,$meta.progress)|Where-Object{Test-Path $_}}else{@()}
$latest=$paths|ForEach-Object{Get-Item $_}|Sort-Object LastWriteTime -Descending|Select-Object -First 1
$gpuPids=@($pidValue)+@($children|ForEach-Object{$_.ProcessId})
$gpuRows=& nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null
$gpu=@($gpuRows|Where-Object{$row=$_;$gpuPids|Where-Object{$row -match "^\s*$_,"}})
$device=& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
Write-Output "status: $(if($process){'RUNNING'}else{'NOT RUNNING'})"
Write-Output "done/total: $(if($progress){$progress.done.ToString()+'/'+$progress.total}else{'0/100'})"
Write-Output "stage: $(if($progress){$progress.stage}else{'not_started'})"
Write-Output "pid: $pidValue"
Write-Output "start_time: $(if($meta){$meta.started}else{'unknown'})"
if($process){Write-Output "command: $($process.CommandLine)"}
if(@($children).Length -gt 0){Write-Output "runtime_pid: $($children[0].ProcessId)";Write-Output "runtime_command: $($children[0].CommandLine)"}else{Write-Output 'runtime_pid: none'}
Write-Output "last_output_timestamp: $(if($latest){$latest.LastWriteTime}else{'none'})"
Write-Output "last_completed_unit: $(if($progress){$progress|ConvertTo-Json -Compress}else{'none'})"
Write-Output "gpu_signal: $(if($gpu){($gpu -join '; ')+'; device='+($device -join '; ')}elseif(@($children).Length -gt 0){'device='+($device -join '; ')}else{'NO GPU SIGNAL YET'})"
Write-Output "stdout: $($meta.stdout)"
Write-Output "stderr: $($meta.stderr)"
Write-Output "progress: $($meta.progress)"

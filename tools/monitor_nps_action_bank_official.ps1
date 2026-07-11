param(
  [string]$RepoRoot='C:\Users\aaron\Desktop\URAP',
  [string]$RunId='nps_action_bank_v1'
)
$root=Join-Path $RepoRoot 'artifacts\detached_nps_action_bank_v1'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile | Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw | ConvertFrom-Json}else{$null}
$progressPath=Join-Path $root 'progress.json'
$progress=if(Test-Path $progressPath){Get-Content $progressPath -Raw | ConvertFrom-Json}else{$null}
$shimInfo=if($progress -and $progress.child_pid){Get-CimInstance Win32_Process -Filter "ProcessId=$($progress.child_pid)" -ErrorAction SilentlyContinue}else{$null}
$runtimeInfo=if($progress -and $progress.child_pid){Get-CimInstance Win32_Process -Filter "ParentProcessId=$($progress.child_pid)" -ErrorAction SilentlyContinue | Select-Object -First 1}else{$null}
$runtime=if($runtimeInfo){Get-Process -Id $runtimeInfo.ProcessId -ErrorAction SilentlyContinue}else{$null}
$lastLine=if($meta -and (Test-Path $meta.stdout)){Get-Content $meta.stdout -Tail 1}else{$null}
$status=if($process){'RUNNING'}else{'NOT RUNNING'}
$startTime=if($process){$process.StartTime}else{if($meta){$meta.started}else{'unknown'}}
$done=if($progress){$progress.done}else{0}
$total=if($progress){$progress.total}else{5}
$stage=if($progress){$progress.stage}else{'not_started'}
$childPid=if($progress){$progress.child_pid}else{'none'}
$childCommand=if($shimInfo){$shimInfo.CommandLine}else{'none'}
$runtimePid=if($runtimeInfo){$runtimeInfo.ProcessId}else{'none'}
$runtimeCommand=if($runtimeInfo){$runtimeInfo.CommandLine}else{'none'}
$runtimeCpu=if($runtime){$runtime.CPU}else{'none'}
$runtimeMemory=if($runtime){[math]::Round($runtime.WorkingSet64/1MB,1)}else{'none'}
$lastTimestamp=if($meta -and (Test-Path $meta.stdout)){(Get-Item $meta.stdout).LastWriteTime}else{'none'}
$gpuSignal=if($lastLine -match 'cuda_memory_allocated_mb'){$lastLine}else{'NO GPU SIGNAL YET (cache/input stage or no completed batch)'}
Write-Output "status: $status"
Write-Output "done/total: $done/$total"
Write-Output "stage: $stage"
Write-Output "pid: $pidValue"
Write-Output "start_time: $startTime"
Write-Output "child_pid: $childPid"
Write-Output "child_command: $childCommand"
Write-Output "runtime_pid: $runtimePid"
Write-Output "runtime_command: $runtimeCommand"
Write-Output "runtime_cpu_seconds: $runtimeCpu"
Write-Output "runtime_working_set_mb: $runtimeMemory"
Write-Output "last_output_timestamp: $lastTimestamp"
Write-Output "last_completed_unit: $lastLine"
Write-Output "gpu_signal: $gpuSignal"
Write-Output "stdout: $($meta.stdout)"
Write-Output "stderr: $($meta.stderr)"
Write-Output "progress: $progressPath"
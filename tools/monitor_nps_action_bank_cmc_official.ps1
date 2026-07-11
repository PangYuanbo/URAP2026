param(
  [string]$RepoRoot='C:\Users\aaron\Desktop\URAP',
  [string]$RunId='nps_action_bank_cmc_v2'
)
$root=Join-Path $RepoRoot 'artifacts\detached_nps_action_bank_cmc_v2'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile | Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw | ConvertFrom-Json}else{$null}
$progressPath=Join-Path $root 'progress.json'
$progress=if(Test-Path $progressPath){Get-Content $progressPath -Raw | ConvertFrom-Json}else{$null}
$childInfo=if($progress -and $progress.child_pid){Get-CimInstance Win32_Process -Filter "ProcessId=$($progress.child_pid)" -ErrorAction SilentlyContinue}else{$null}
$runtimeInfo=if($progress -and $progress.child_pid){Get-CimInstance Win32_Process -Filter "ParentProcessId=$($progress.child_pid)" -ErrorAction SilentlyContinue | Select-Object -First 1}else{$null}
$runtime=if($runtimeInfo){Get-Process -Id $runtimeInfo.ProcessId -ErrorAction SilentlyContinue}else{$null}
$childProgress=if($progress -and $progress.child_progress -and (Test-Path $progress.child_progress)){Get-Content $progress.child_progress -Raw | ConvertFrom-Json}else{$null}
$status=if($process){'RUNNING'}else{'NOT RUNNING'}
$startTime=if($process){$process.StartTime}else{if($meta){$meta.started}else{'unknown'}}
$done=if($progress){$progress.done}else{0}
$total=if($progress){$progress.total}else{10}
$stage=if($progress){$progress.stage}else{'not_started'}
$lastLine=if($meta -and (Test-Path $meta.stdout)){Get-Content $meta.stdout -Tail 1}else{'none'}
$gpuLine=if($meta -and (Test-Path $meta.stdout)){Get-Content $meta.stdout -Tail 200 | Select-String 'cuda_memory_allocated_mb' | Select-Object -Last 1}else{$null}
$gpuSignal=if($gpuLine){$gpuLine.Line}else{'NO GPU SIGNAL YET (CPU CMC/cache stage or no completed CUDA batch)'}
$lastTimestamp=if($meta -and (Test-Path $meta.stdout)){(Get-Item $meta.stdout).LastWriteTime}else{'none'}
Write-Output "status: $status"
Write-Output "done/total: $done/$total"
Write-Output "stage: $stage"
Write-Output "pid: $pidValue"
Write-Output "start_time: $startTime"
Write-Output "child_pid: $(if($progress){$progress.child_pid}else{'none'})"
Write-Output "child_command: $(if($childInfo){$childInfo.CommandLine}else{'none'})"
Write-Output "runtime_pid: $(if($runtimeInfo){$runtimeInfo.ProcessId}else{'none'})"
Write-Output "runtime_command: $(if($runtimeInfo){$runtimeInfo.CommandLine}else{'none'})"
Write-Output "runtime_cpu_seconds: $(if($runtime){$runtime.CPU}else{'none'})"
Write-Output "runtime_working_set_mb: $(if($runtime){[math]::Round($runtime.WorkingSet64/1MB,1)}else{'none'})"
Write-Output "child_done/total: $(if($childProgress){$childProgress.done.ToString()+'/'+$childProgress.total}else{'none'})"
Write-Output "last_output_timestamp: $lastTimestamp"
Write-Output "last_completed_unit: $lastLine"
Write-Output "gpu_signal: $gpuSignal"
Write-Output "stdout: $($meta.stdout)"
Write-Output "stderr: $($meta.stderr)"
Write-Output "progress: $progressPath"

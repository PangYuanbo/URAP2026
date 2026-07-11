param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_official_val_to_test_cuda_rank_v2')
$root=Join-Path $RepoRoot 'artifacts\detached_nps_official_val_to_test_cuda_rank'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$progress=if(Test-Path (Join-Path $root 'progress.json')){Get-Content (Join-Path $root 'progress.json') -Raw|ConvertFrom-Json}else{$null}
$child=if($progress.child_pid){Get-CimInstance Win32_Process -Filter "ProcessId=$($progress.child_pid)" -ErrorAction SilentlyContinue}else{$null}
$gpu=& nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null
[ordered]@{running=[bool]$process;pid=$pidValue;command_line=if($process){(Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue").CommandLine}else{$null};start_time=if($process){$process.StartTime}else{$null};done=if($progress){$progress.done}else{0};total=if($progress){$progress.total}else{4};stage=if($progress){$progress.stage}else{'launch'};child_pid=if($progress){$progress.child_pid}else{$null};child_command=if($child){$child.CommandLine}else{$null};last_completed_unit=if($meta-and(Test-Path $meta.stdout)){Get-Content $meta.stdout -Tail 1}else{$null};last_output_timestamp=if($meta-and(Test-Path $meta.stdout)){(Get-Item $meta.stdout).LastWriteTime}else{$null};gpu_processes=$gpu;stdout=if($meta){$meta.stdout}else{$null};stderr=if($meta){$meta.stderr}else{$null}}|ConvertTo-Json -Depth 8

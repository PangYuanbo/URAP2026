param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_visual_crop_score_v1')

$root=Join-Path $RepoRoot 'artifacts\detached_nps_visual_crop_score_eval'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$progress=if($meta-and(Test-Path $meta.progress)){Get-Content $meta.progress -Raw|ConvertFrom-Json}else{$null}
$scoreProgress=if($meta-and(Test-Path $meta.score_progress)){Get-Content $meta.score_progress -Raw|ConvertFrom-Json}else{$null}
$child=if($progress.child_pid){Get-CimInstance Win32_Process -Filter "ProcessId=$($progress.child_pid)" -ErrorAction SilentlyContinue}else{$null}
$gpu=& nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null
[ordered]@{running=[bool]$process;pid=$pidValue;command_line=if($process){(Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue").CommandLine}else{$null};start_time=if($process){$process.StartTime}else{$null};done=if($scoreProgress-and$progress.stage-eq'score'){$scoreProgress.done}elseif($progress){$progress.done}else{0};total=if($scoreProgress-and$progress.stage-eq'score'){$scoreProgress.total}elseif($progress){$progress.total}else{2};stage=if($progress){$progress.stage}else{'launch'};child_pid=if($progress){$progress.child_pid}else{$null};child_command=if($child){$child.CommandLine}else{$null};last_completed_unit=if($meta-and(Test-Path $meta.stdout)){Get-Content $meta.stdout -Tail 1}else{$null};last_output_timestamp=if($meta-and(Test-Path $meta.stdout)){(Get-Item $meta.stdout).LastWriteTime}else{$null};gpu=$gpu;stdout=if($meta){$meta.stdout}else{$null};stderr=if($meta){$meta.stderr}else{$null}}|ConvertTo-Json -Depth 6

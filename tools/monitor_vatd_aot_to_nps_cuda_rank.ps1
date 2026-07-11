param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='vatd_aot_to_nps_cuda_rank_v1')
$root=Join-Path $RepoRoot 'artifacts\detached_vatd_aot_to_nps_cuda_rank';$pidFile=Join-Path $root ($RunId+'.pid');$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0};$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null};$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$status=[ordered]@{running=[bool]$process;pid=$pidValue;start_time=if($process){$process.StartTime}else{$null};stdout=if($meta){$meta.stdout}else{$null};stderr=if($meta){$meta.stderr}else{$null};last_log_time=$null;last_log_line=$null;done=0;total=2;stage='launch'}
if($meta -and (Test-Path $meta.stdout)){$file=Get-Item $meta.stdout;$status.last_log_time=$file.LastWriteTime;$status.last_log_line=Get-Content $meta.stdout -Tail 1}
try{$remote=modal volume get vatd-rank-results-v1 /aot_to_nps_cuda_rank_v1/progress.json - 2>$null;if($remote){$progress=$remote|ConvertFrom-Json;$status.done=$progress.done;$status.total=$progress.total;$status.stage=$progress.stage;$status.remote_progress=$progress}}catch{}
$status|ConvertTo-Json -Depth 8

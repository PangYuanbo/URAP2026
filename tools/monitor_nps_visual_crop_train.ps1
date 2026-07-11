param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_visual_crop_v1')

$root=Join-Path $RepoRoot 'artifacts\detached_nps_visual_crop_train'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$last=if($meta-and(Test-Path $meta.stdout)){Get-Content $meta.stdout -Tail 1}else{$null}
$epoch=0;if($last -and $last -match '"epoch"\s*:\s*(\d+)'){$epoch=[int]$Matches[1]}
$gpu=& nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null
[ordered]@{running=[bool]$process;pid=$pidValue;command_line=if($process){(Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue").CommandLine}else{$null};start_time=if($process){$process.StartTime}else{$null};done=$epoch;total=6;stage=if($process){'train'}elseif($meta-and(Test-Path $meta.summary)){'done'}else{'stopped'};last_completed_unit=$last;last_output_timestamp=if($meta-and(Test-Path $meta.stdout)){(Get-Item $meta.stdout).LastWriteTime}else{$null};gpu=$gpu;stdout=if($meta){$meta.stdout}else{$null};stderr=if($meta){$meta.stderr}else{$null}}|ConvertTo-Json -Depth 5

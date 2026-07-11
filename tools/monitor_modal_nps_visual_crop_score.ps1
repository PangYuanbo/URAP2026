param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='modal_nps_visual_crop_score_v1')

$root=Join-Path $RepoRoot 'artifacts\detached_modal_nps_visual_crop_score'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$line=if($meta-and(Test-Path $meta.stdout)){[string](Get-Content $meta.stdout -Tail 1)}else{$null}
$parsed=$null;try{$parsed=$line|ConvertFrom-Json -ErrorAction Stop}catch{}
[ordered]@{running=[bool]$process;pid=$pidValue;command_line=if($process){(Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue").CommandLine}else{$null};start_time=if($process){$process.StartTime}else{$null};done=if($parsed-and$parsed.done){$parsed.done}else{0};total=if($parsed-and$parsed.total){$parsed.total}else{12350};stage=if($parsed-and$parsed.stage){$parsed.stage}elseif($process){'build_or_launch'}else{'stopped'};last_completed_unit=$line;last_output_timestamp=if($meta-and(Test-Path $meta.stdout)){(Get-Item $meta.stdout).LastWriteTime}else{$null};stdout=if($meta){$meta.stdout}else{$null};stderr=if($meta){$meta.stderr}else{$null}}|ConvertTo-Json -Depth 5

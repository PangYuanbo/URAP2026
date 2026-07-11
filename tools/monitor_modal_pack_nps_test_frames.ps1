param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_test_frames_pack_v1')

$root=Join-Path $RepoRoot 'artifacts\detached_nps_test_frames_pack'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$last=if($meta-and(Test-Path $meta.stdout)){Get-Content $meta.stdout -Tail 1}else{$null}
[ordered]@{running=[bool]$process;pid=$pidValue;command_line=if($process){(Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue").CommandLine}else{$null};start_time=if($process){$process.StartTime}else{$null};done=if($last-match'archive_bytes'){1}else{0};total=1;stage=if($process){'pack'}elseif($last-match'archive_bytes'){'done'}else{'stopped'};last_completed_unit=$last;last_output_timestamp=if($meta-and(Test-Path $meta.stdout)){(Get-Item $meta.stdout).LastWriteTime}else{$null};stdout=if($meta){$meta.stdout}else{$null};stderr=if($meta){$meta.stderr}else{$null}}|ConvertTo-Json -Depth 5

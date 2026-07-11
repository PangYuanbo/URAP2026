param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_test_frames_archive_v1')

$root=Join-Path $RepoRoot 'artifacts\detached_nps_test_frames_archive_download'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$archive=if($meta){$meta.archive}else{'D:\URAP_nps_test_pack\NPS_AllFrames_test.tar'}
[ordered]@{running=[bool]$process;pid=$pidValue;command_line=if($process){(Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue").CommandLine}else{$null};start_time=if($process){$process.StartTime}else{$null};done=if(Test-Path $archive){(Get-Item $archive).Length}else{0};total='2.9 GiB visible remote';stage=if($process){'download'}elseif(Test-Path $archive){'done'}else{'stopped'};last_output_timestamp=if(Test-Path $archive){(Get-Item $archive).LastWriteTime}elseif($meta-and(Test-Path $meta.stdout)){(Get-Item $meta.stdout).LastWriteTime}else{$null};stdout=if($meta){$meta.stdout}else{$null};stderr=if($meta){$meta.stderr}else{$null}}|ConvertTo-Json -Depth 5

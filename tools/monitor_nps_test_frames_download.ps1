param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_test_frames_v1')

$root=Join-Path $RepoRoot 'artifacts\detached_nps_test_frames_download'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$progress=if($meta -and (Test-Path $meta.progress)){Get-Content $meta.progress -Raw|ConvertFrom-Json}else{$null}
$files=if(Test-Path 'D:\URAP_nps_test_tvd\AllFrames'){Get-ChildItem 'D:\URAP_nps_test_tvd\AllFrames' -Recurse -File -Filter '*.png' -ErrorAction SilentlyContinue}else{@()}
[ordered]@{running=[bool]$process;pid=$pidValue;command_line=if($process){(Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue").CommandLine}else{$null};start_time=if($process){$process.StartTime}else{$null};done=$files.Count;total=12355;stage=if($progress){$progress.stage}else{'launch'};last_file=if($files.Count){($files|Sort-Object LastWriteTime|Select-Object -Last 1).FullName}else{$null};last_output_timestamp=if($files.Count){($files|Sort-Object LastWriteTime|Select-Object -Last 1).LastWriteTime}else{$null};stdout=if($meta){$meta.stdout}else{$null};stderr=if($meta){$meta.stderr}else{$null}}|ConvertTo-Json -Depth 5

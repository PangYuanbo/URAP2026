param()
$ErrorActionPreference="Stop"
$runDir="C:\Users\aaron\Desktop\URAP\artifacts\detached_ard100_generalization_prepare_v1"
$meta=Get-Content (Join-Path $runDir "meta.json") -Raw|ConvertFrom-Json
$progressPath=Join-Path $runDir "progress.json"
$progress=if(Test-Path $progressPath){Get-Content $progressPath -Raw|ConvertFrom-Json}else{[pscustomobject]@{stage="starting";done=0;total=4;updated=$null}}
$all=Get-CimInstance Win32_Process; $root=$all|Where-Object {$_.ProcessId -eq [int]$meta.pid}; $children=@($all|Where-Object {$_.ParentProcessId -eq [int]$meta.pid}); $matching=@($root)+$children|Where-Object {$_ -and ($_.CommandLine -match "run_ard100_generalization_prepare|merge_ard100|eval_tvd|precompute_nps_homographies")}; $alive=$matching.Count -gt 0
$last=if(Test-Path $progressPath){(Get-Item $progressPath).LastWriteTime.ToString("o")}elseif(Test-Path $meta.stdout_log){(Get-Item $meta.stdout_log).LastWriteTime.ToString("o")}else{$null}
[pscustomobject]@{status=if($alive){"RUNNING"}else{"NOT RUNNING"};done=[int]$progress.done;total=[int]$progress.total;stage=$progress.stage;launcher_pid=[int]$meta.pid;process_pids=@($matching|Select-Object -ExpandProperty ProcessId -Unique);start_time=$meta.start_time;last_output_timestamp=$last;last_completed_unit=$progress.stage;stdout_log=$meta.stdout_log;stderr_log=$meta.stderr_log;progress_json=$progressPath}|ConvertTo-Json -Depth 3

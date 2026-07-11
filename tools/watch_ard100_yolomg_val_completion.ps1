param([int]$DetectorPid=59324)
$ErrorActionPreference='Stop'
$run='C:\Users\aaron\Desktop\URAP\artifacts\detached_ard100_yolomg_val_candidates_v1'
$meta=Get-Content (Join-Path $run 'meta.json') -Raw|ConvertFrom-Json
$marker=Join-Path $meta.output_dir 'results.txt';$progress=Join-Path $run 'completion_watcher.json'
while(Get-Process -Id $DetectorPid -ErrorAction SilentlyContinue){@{status='waiting';pid=$DetectorPid;updated=(Get-Date).ToString('o')}|ConvertTo-Json|Set-Content $progress -Encoding UTF8;Start-Sleep 20}
$log=Get-Content $meta.stderr_log -Raw;$labels=(Get-ChildItem (Join-Path $meta.output_dir 'labels') -File -Filter '*.txt' -ErrorAction SilentlyContinue).Count
if($log -notmatch 'Results saved to'){@{status='stopped_without_completion';pid=$DetectorPid;labels=$labels;updated=(Get-Date).ToString('o')}|ConvertTo-Json|Set-Content $progress -Encoding UTF8;exit 1}
@{status='verified_complete';pid=$DetectorPid;labels=$labels;verified_at=(Get-Date).ToString('o');stderr_log=$meta.stderr_log}|ConvertTo-Json|Set-Content $marker -Encoding UTF8
@{status='done';pid=$DetectorPid;labels=$labels;marker=$marker;updated=(Get-Date).ToString('o')}|ConvertTo-Json|Set-Content $progress -Encoding UTF8

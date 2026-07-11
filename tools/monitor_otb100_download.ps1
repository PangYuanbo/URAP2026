param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='otb100_download')
$root=Join-Path $RepoRoot 'artifacts\detached_otb100_download'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$target=if($meta){$meta.target}else{'D:\URAP_local_datasets\OTB100'}
$sequences=if(Test-Path $target){@(Get-ChildItem $target -Directory -ErrorAction SilentlyContinue).Count}else{0}
$images=if(Test-Path $target){@(Get-ChildItem $target -Recurse -File -Filter '*.jpg' -ErrorAction SilentlyContinue).Count}else{0}
$lastFile=if(Test-Path $target){Get-ChildItem $target -Recurse -File -ErrorAction SilentlyContinue|Sort-Object LastWriteTime -Descending|Select-Object -First 1}else{$null}
Write-Output "status: $(if($process){'RUNNING'}else{'NOT RUNNING'})"
Write-Output "done/total: $sequences/100_sequences"
Write-Output "pid: $pidValue"
Write-Output "start_time: $(if($process){$process.StartTime}else{if($meta){$meta.started}else{$null}})"
Write-Output "last_output_timestamp: $(if($lastFile){$lastFile.LastWriteTime}else{$null})"
Write-Output "last_completed_unit: sequences=$sequences images=$images last_file=$(if($lastFile){$lastFile.FullName}else{$null})"
Write-Output "stdout: $(if($meta){$meta.stdout}else{$null})"
Write-Output "stderr: $(if($meta){$meta.stderr}else{$null})"
Write-Output "target: $target"

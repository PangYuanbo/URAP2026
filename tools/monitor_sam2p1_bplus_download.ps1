param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='sam2p1_bplus_download')
$root=Join-Path $RepoRoot 'artifacts\detached_sam2p1_bplus_download'
$pidFile=Join-Path $root ($RunId+'.pid')
$metaFile=Join-Path $root ($RunId+'.meta.json')
$pidValue=if(Test-Path $pidFile){[int](Get-Content $pidFile|Select-Object -First 1)}else{0}
$process=if($pidValue){Get-Process -Id $pidValue -ErrorAction SilentlyContinue}else{$null}
$meta=if(Test-Path $metaFile){Get-Content $metaFile -Raw|ConvertFrom-Json}else{$null}
$bytes=if($meta -and (Test-Path $meta.target)){(Get-Item $meta.target).Length}else{0}
$lastTimestamp=if($meta -and (Test-Path $meta.target)){(Get-Item $meta.target).LastWriteTime}else{$null}
Write-Output "status: $(if($process){'RUNNING'}else{'NOT RUNNING'})"
Write-Output "done/total: $bytes/unknown_bytes"
Write-Output "pid: $pidValue"
Write-Output "start_time: $(if($process){$process.StartTime}else{if($meta){$meta.started}else{$null}})"
Write-Output "command: $(if($meta){$meta.command -join ' '}else{$null})"
Write-Output "last_output_timestamp: $lastTimestamp"
Write-Output "last_completed_unit: downloaded_bytes=$bytes"
Write-Output "stdout: $(if($meta){$meta.stdout}else{$null})"
Write-Output "stderr: $(if($meta){$meta.stderr}else{$null})"
Write-Output "target: $(if($meta){$meta.target}else{$null})"

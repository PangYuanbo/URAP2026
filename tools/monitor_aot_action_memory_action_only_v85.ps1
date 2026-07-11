$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Run=Join-Path $Repo 'artifacts\detached_aot_action_memory_action_only_v85'
$StatePath=Join-Path $Run 'state.json'
if(-not(Test-Path $StatePath)){Write-Host 'status: NOT RUNNING';Write-Host 'done/total: 0/5';exit 0}
$State=Get-Content $StatePath -Raw|ConvertFrom-Json
$Process=Get-CimInstance Win32_Process -Filter "ProcessId=$($State.pid)" -ErrorAction SilentlyContinue
$Progress=if(Test-Path $State.progress){Get-Content $State.progress -Raw|ConvertFrom-Json}else{$null}
$ChildId=if($Progress -and $Progress.child_pid){[int]$Progress.child_pid}else{0}
$All=@(Get-CimInstance Win32_Process);$Desc=@();$Front=@($ChildId)|Where-Object{$_ -gt 0}
while($Front.Count){$Next=@();foreach($ParentId in $Front){$Found=@($All|Where-Object{$_.ParentProcessId -eq $ParentId});$Desc+=$Found;$Next+=@($Found|ForEach-Object{$_.ProcessId})};$Front=$Next}
$Paths=@($State.stdout,$State.stderr,$State.progress)|Where-Object{Test-Path $_};$Latest=$Paths|ForEach-Object{Get-Item $_}|Sort-Object LastWriteTime -Descending|Select-Object -First 1
Write-Host ("status: "+$(if($Process){'RUNNING'}else{'NOT RUNNING'}));Write-Host ("done/total: "+$(if($Progress){"$($Progress.done)/$($Progress.total)"}else{'0/5'}));Write-Host ("stage: "+$(if($Progress){$Progress.stage}else{'starting'}));Write-Host "pid: $($State.pid)";Write-Host "start_time: $($State.start_time)"
if($Process){Write-Host "command: $($Process.CommandLine)"};Write-Host "child_pid: $ChildId";foreach($Runtime in $Desc){Write-Host "runtime_pid: $($Runtime.ProcessId)";Write-Host "runtime_working_set_mb: $([math]::Round($Runtime.WorkingSetSize/1MB,1))"}
Write-Host ("last_output_timestamp: "+$(if($Latest){$Latest.LastWriteTime}else{'none'}));Write-Host ("last_completed_unit: "+$(if($Progress){$Progress|ConvertTo-Json -Compress}else{'none'}))
if($Progress -and $Progress.stage -eq 'build_camera_compensated_action_memory' -and (Test-Path $State.stdout)){
  $CompletedSequences=(Get-Content $State.stdout|Select-String '"kind": "online_action_bank_sequence"').Count
  Write-Host "action_memory_sequences: $CompletedSequences/172"
}
$GpuPids=@($State.pid,$ChildId)+@($Desc|ForEach-Object{$_.ProcessId});$GpuRows=& nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null;$Gpu=@($GpuRows|Where-Object{$row=$_;$GpuPids|Where-Object{$row -match "^\s*$_,"}});$Device=& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
Write-Host ("gpu_signal: "+$(if($Gpu){($Gpu -join '; ')+'; device='+($Device -join '; ')}elseif($ChildId){'device='+($Device -join '; ')}else{'none'}));Write-Host "stdout: $($State.stdout)";Write-Host "stderr: $($State.stderr)";Write-Host "progress: $($State.progress)"

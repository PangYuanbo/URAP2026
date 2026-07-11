$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Run=Join-Path $Repo 'artifacts\detached_action_chunk_dual_memory_v56'
$StatePath=Join-Path $Run 'state.json'
if(-not(Test-Path $StatePath)){Write-Host 'status: NOT RUNNING';Write-Host 'done/total: 0/3';exit 0}
$State=Get-Content $StatePath -Raw|ConvertFrom-Json
$Process=Get-CimInstance Win32_Process -Filter "ProcessId=$($State.pid)" -ErrorAction SilentlyContinue
$Progress=if(Test-Path $State.progress){Get-Content $State.progress -Raw|ConvertFrom-Json}else{$null}
$ChildId=if($Progress -and $Progress.child_pid){[int]$Progress.child_pid}else{0}
$Child=if($ChildId){Get-CimInstance Win32_Process -Filter "ProcessId=$ChildId" -ErrorAction SilentlyContinue}else{$null}
$Runtime=if($ChildId){@(Get-CimInstance Win32_Process|Where-Object{$_.ParentProcessId -eq $ChildId})}else{@()}
$Paths=@($State.stdout,$State.stderr,$State.progress)|Where-Object{Test-Path $_}
$Latest=$Paths|ForEach-Object{Get-Item $_}|Sort-Object LastWriteTime -Descending|Select-Object -First 1
Write-Host ("status: "+$(if($Process){'RUNNING'}else{'NOT RUNNING'}))
Write-Host ("done/total: "+$(if($Progress){"$($Progress.done)/$($Progress.total)"}else{'0/3'}))
Write-Host ("stage: "+$(if($Progress){$Progress.stage}else{'starting'}))
Write-Host "pid: $($State.pid)"
Write-Host "start_time: $($State.start_time)"
if($Process){Write-Host "command: $($Process.CommandLine)"}
if($Child){Write-Host "child_pid: $($Child.ProcessId)";Write-Host "child_command: $($Child.CommandLine)"}else{Write-Host 'child_pid: none'}
if(@($Runtime).Length){Write-Host "runtime_pid: $($Runtime[0].ProcessId)";Write-Host "runtime_command: $($Runtime[0].CommandLine)"}else{Write-Host 'runtime_pid: none'}
Write-Host ("last_output_timestamp: "+$(if($Latest){$Latest.LastWriteTime}else{'none'}))
Write-Host ("last_completed_unit: "+$(if($Progress){$Progress|ConvertTo-Json -Compress}else{'none'}))
$GpuPids=@($State.pid,$ChildId)+@($Runtime|ForEach-Object{$_.ProcessId})
$GpuRows=& nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>$null
$Gpu=@($GpuRows|Where-Object{$row=$_;$GpuPids|Where-Object{$row -match "^\s*$_,"}})
$Device=& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null
Write-Host ("gpu_signal: "+$(if($Gpu){($Gpu -join '; ')+'; device='+($Device -join '; ')}elseif($Child){'device='+($Device -join '; ')}else{'none'}))
Write-Host "stdout: $($State.stdout)"
Write-Host "stderr: $($State.stderr)"
Write-Host "progress: $($State.progress)"

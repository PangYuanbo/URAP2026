$ErrorActionPreference='Stop'
$Repo=Split-Path -Parent $PSScriptRoot;$Run=Join-Path $Repo 'artifacts\detached_tvd_samurai_memory_v93';$PidFile=Join-Path $Run 'pid.txt';$ProgressFile=Join-Path $Run 'progress.json';$MetaFile=Join-Path $Run 'start_meta.json'
if(!(Test-Path $PidFile)){Write-Output 'NOT RUNNING: no PID file';exit 1}
$JobPid=[int](Get-Content $PidFile -Raw).Trim();$Process=Get-CimInstance Win32_Process -Filter "ProcessId=$JobPid" -ErrorAction SilentlyContinue;$Meta=Get-Content $MetaFile -Raw|ConvertFrom-Json
if($Process){Write-Output "RUNNING PID=$JobPid START=$($Meta.start_time)";Write-Output "COMMAND=$($Process.CommandLine)"}else{Write-Output "NOT RUNNING PID=$JobPid START=$($Meta.start_time)"}
if(Test-Path $ProgressFile){$Progress=Get-Content $ProgressFile -Raw|ConvertFrom-Json;Write-Output "PROGRESS=$($Progress.done)/$($Progress.total) STAGE=$($Progress.stage) UPDATED=$($Progress.updated) CHILD_PID=$($Progress.child_pid)";if($Progress.child_pid){$Child=Get-CimInstance Win32_Process -Filter "ProcessId=$($Progress.child_pid)" -ErrorAction SilentlyContinue;Write-Output "CHILD_ALIVE=$([bool]$Child)"}}
foreach($Name in @('stdout.log','stderr.log')){$Log=Join-Path $Run $Name;if(Test-Path $Log){$Item=Get-Item $Log;Write-Output "LOG=$Log LAST_WRITE=$($Item.LastWriteTime.ToString('o')) BYTES=$($Item.Length)";Get-Content $Log -Tail 15}}
& nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits

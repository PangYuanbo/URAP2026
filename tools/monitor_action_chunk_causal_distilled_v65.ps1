param()
$Repo='C:\Users\aaron\Desktop\URAP'
$Run=Join-Path $Repo 'artifacts\detached_action_chunk_causal_distilled_v65'
$Job=Get-Content (Join-Path $Run 'job.json') -Raw|ConvertFrom-Json
$All=Get-CimInstance Win32_Process
$Ids=@([int]$Job.pid)
$Changed=$true
while($Changed){$Changed=$false;foreach($Item in $All){if($Ids -contains [int]$Item.ParentProcessId -and -not($Ids -contains [int]$Item.ProcessId)){$Ids += [int]$Item.ProcessId;$Changed=$true}}}
$Alive=@($All|Where-Object{$Ids -contains [int]$_.ProcessId -and $_.Name -match 'python'})
$ProgressPath=Join-Path $Run 'progress.json'
$Progress=if(Test-Path $ProgressPath){Get-Content $ProgressPath -Raw|ConvertFrom-Json}else{$null}
if($Alive.Count){'status: RUNNING'}else{'status: NOT RUNNING'}
if($Progress){"done/total: $($Progress.done)/$($Progress.total)";"stage: $($Progress.stage)"}else{'done/total: 0/3';'stage: starting'}
"pid: $($Job.pid)";"start_time: $($Job.start_time)"
foreach($Item in $Alive){$Native=Get-Process -Id $Item.ProcessId -ErrorAction SilentlyContinue;if($Native){"runtime: PID=$($Item.ProcessId) CPU=$([math]::Round($Native.CPU,2)) WorkingMB=$([math]::Round($Native.WorkingSet64/1MB,1)) PrivateMB=$([math]::Round($Native.PrivateMemorySize64/1MB,1))"}}
$Log=Get-Item $Job.stdout -ErrorAction SilentlyContinue;if($Log){"last_output_timestamp: $($Log.LastWriteTime)";Get-Content $Job.stdout -Tail 1 -ErrorAction SilentlyContinue}
"stdout: $($Job.stdout)";"stderr: $($Job.stderr)"

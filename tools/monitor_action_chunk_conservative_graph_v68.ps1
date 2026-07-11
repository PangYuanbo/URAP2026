param()
$Repo='C:\Users\aaron\Desktop\URAP'
$Run=Join-Path $Repo 'artifacts\detached_action_chunk_conservative_graph_v68'
$Job=Get-Content (Join-Path $Run 'job.json') -Raw|ConvertFrom-Json
$Process=Get-CimInstance Win32_Process -Filter "ProcessId=$($Job.pid)" -ErrorAction SilentlyContinue
$ProgressPath=Join-Path $Run 'progress.json'
$Progress=if(Test-Path $ProgressPath){Get-Content $ProgressPath -Raw|ConvertFrom-Json}else{$null}
if($Process){'status: RUNNING'}else{'status: NOT RUNNING'}
if($Progress){"done/total: $($Progress.done)/$($Progress.total)";"stage: $($Progress.stage)";"last_output_timestamp: $((Get-Item $ProgressPath).LastWriteTime)"}else{'done/total: 0/4';'stage: starting'}
"pid: $($Job.pid)";"start_time: $($Job.start_time)";"stdout: $($Job.stdout)";"stderr: $($Job.stderr)"
if($Progress.child_pid){$Child=Get-CimInstance Win32_Process -Filter "ProcessId=$($Progress.child_pid)" -ErrorAction SilentlyContinue;"child_pid: $($Progress.child_pid)";if($Child){"child_command: $($Child.CommandLine)"}}

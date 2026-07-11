param([string]$RepoRoot = "C:\Users\aaron\Desktop\URAP")

$runnerDir=Join-Path $RepoRoot "artifacts\modal_ard100_raw_upload"
$pidPath=Join-Path $runnerDir "upload.pid"
$metaPath=Join-Path $runnerDir "upload.meta.txt"
$progressPath=Join-Path $runnerDir "progress.json"
$pidValue=if(Test-Path $pidPath){(Get-Content $pidPath -Raw).Trim()}else{""}
$process=if($pidValue -match "^\d+$"){Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue}else{$null}
if($process){Write-Host "RUNNING=true PID=$pidValue";Write-Host "PROCESS_COMMAND=$($process.CommandLine)"}else{Write-Host "NOT RUNNING PID=$pidValue"}
if(Test-Path $progressPath){$p=Get-Content $progressPath -Raw|ConvertFrom-Json;Write-Host "done/total=$($p.done)/$($p.total) status=$($p.status) current=$($p.current) updated=$($p.updated)"}
if(Test-Path $metaPath){$m=@{};foreach($line in Get-Content $metaPath){if($line -match "^([^=]+)=(.*)$"){$m[$Matches[1]]=$Matches[2]}};Write-Host "start time: $($m.started)";Write-Host "stdout log: $($m.stdout)";Write-Host "stderr log: $($m.stderr)";if(Test-Path $m.stdout){Get-Content $m.stdout -Tail 8};if(Test-Path $m.stderr){Get-Content $m.stderr -Tail 8}}

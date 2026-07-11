$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Python=Join-Path $Repo 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$Run=Join-Path $Repo 'artifacts\detached_nps_teacher_motion_token_gate_v16'
$Logs=Join-Path $Run 'logs'
New-Item -ItemType Directory -Force $Logs|Out-Null
$State=Join-Path $Run 'state.json'
if(Test-Path $State){$old=Get-Content $State -Raw|ConvertFrom-Json;$proc=Get-CimInstance Win32_Process -Filter "ProcessId=$($old.pid)" -ErrorAction SilentlyContinue;if($proc -and $proc.CommandLine -like '*run_nps_teacher_motion_token_gate_v16.py*'){throw "run already active pid=$($old.pid)"}}
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Stdout=Join-Path $Logs "nps_teacher_motion_token_gate_v16_$Stamp.out.txt"
$Stderr=Join-Path $Logs "nps_teacher_motion_token_gate_v16_$Stamp.err.txt"
$Process=Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList @((Join-Path $Repo 'tools\run_nps_teacher_motion_token_gate_v16.py')) -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
@{pid=$Process.Id;start_time=(Get-Date).ToString('o');stdout=$Stdout;stderr=$Stderr;progress=(Join-Path $Run 'progress.json');total=3}|ConvertTo-Json|Set-Content -Encoding UTF8 $State
$Process.Id|Set-Content -Encoding ASCII (Join-Path $Run 'pid.txt')
Write-Host "started pid=$($Process.Id) stdout=$Stdout stderr=$Stderr"

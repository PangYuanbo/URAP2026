param()
$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Run=Join-Path $Repo 'artifacts\detached_action_chunk_frame_presence_v69'
$Logs=Join-Path $Run 'logs'
New-Item -ItemType Directory -Force -Path $Logs|Out-Null
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Out=Join-Path $Logs "action_chunk_frame_presence_v69_$Stamp.out.txt"
$Err=Join-Path $Logs "action_chunk_frame_presence_v69_$Stamp.err.txt"
$Python=Join-Path $Repo 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$Process=Start-Process -FilePath $Python -ArgumentList (Join-Path $Repo 'tools\run_action_chunk_frame_presence_v69.py') -WorkingDirectory $Repo -RedirectStandardOutput $Out -RedirectStandardError $Err -WindowStyle Hidden -PassThru
@{pid=$Process.Id;start_time=(Get-Date).ToString('o');command="$Python tools\run_action_chunk_frame_presence_v69.py";stdout=$Out;stderr=$Err}|ConvertTo-Json|Set-Content (Join-Path $Run 'job.json')
$Process.Id

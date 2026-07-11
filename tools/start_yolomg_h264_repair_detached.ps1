param([int]$MaxWorkers=2)
$ErrorActionPreference='Stop'
$repo=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$latestGeneration=Get-Content (Join-Path $repo 'artifacts\runs\yolomg_pure_1080p_new_batch_latest.json') -Raw|ConvertFrom-Json
$python=Join-Path $repo 'artifacts\venvs\nps_flow\Scripts\python.exe'
$script=Join-Path $repo 'tools\repair_yolomg_h264_batch.py'
$ffmpeg=(Get-ChildItem (Join-Path $repo 'artifacts\tools\ffmpeg-btbn-20260531') -Recurse -Filter ffmpeg.exe -File|Select-Object -First 1).FullName
$ffprobe=(Get-ChildItem (Join-Path $repo 'artifacts\tools\ffmpeg-btbn-20260531') -Recurse -Filter ffprobe.exe -File|Select-Object -First 1).FullName
$runId=Get-Date -Format 'yyyyMMdd_HHmmss';$runDir=Join-Path $repo "artifacts\runs\yolomg_h264_repair\$runId"
$stdout=Join-Path $runDir 'coordinator_stdout.log';$stderr=Join-Path $runDir 'coordinator_stderr.log';$latestFile=Join-Path $repo 'artifacts\runs\yolomg_h264_repair_latest.json'
New-Item -ItemType Directory -Force -Path $runDir|Out-Null
$args=@($script,'--ffmpeg',$ffmpeg,'--ffprobe',$ffprobe,'--generation-status',$latestGeneration.status_json,'--generation-pid',$latestGeneration.coordinator_pid.ToString(),'--old-dir',(Join-Path $repo 'artifacts\yolomg_pure_difference_1080p'),'--result-dir',(Join-Path $repo 'artifacts\yolomg_pure_difference_1080p_new_batch'),'--source-dir','C:\Users\aaron\Desktop\Drone_Videos_Chronological','--backup-dir',(Join-Path $repo 'artifacts\yolomg_unplayable_mp4v_backup'),'--run-dir',$runDir,'--max-workers',$MaxWorkers.ToString())
$process=Start-Process -FilePath $python -ArgumentList $args -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id|Set-Content (Join-Path $runDir 'coordinator_pid.txt') -Encoding ascii
[ordered]@{run_id=$runId;run_dir=$runDir;coordinator_pid=$process.Id;started_at=(Get-Date).ToString('o');stdout_log=$stdout;stderr_log=$stderr;status_json=(Join-Path $runDir 'status.json');result_dir=(Join-Path $repo 'artifacts\yolomg_pure_difference_1080p_new_batch');backup_dir=(Join-Path $repo 'artifacts\yolomg_unplayable_mp4v_backup')}|ConvertTo-Json|Set-Content $latestFile -Encoding utf8
Write-Output "Started H.264 repair coordinator PID=$($process.Id)";Write-Output "Run=$runDir";Write-Output "Logs=$stdout ; $stderr"

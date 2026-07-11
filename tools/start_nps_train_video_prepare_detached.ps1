param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_train_video_prepare_v6')
$ErrorActionPreference='Stop';$root=Join-Path $RepoRoot 'artifacts\detached_nps_train_video_prepare';$logs=Join-Path $root 'logs';New-Item -ItemType Directory -Force -Path $logs|Out-Null;$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss';$stdout=Join-Path $logs ($RunId+'_'+$timestamp+'.out.txt');$stderr=Join-Path $logs ($RunId+'_'+$timestamp+'.err.txt');$python='C:\Users\aaron\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe';$script=Join-Path $RepoRoot 'tools\prepare_nps_train_from_videos.py';$process=Start-Process -FilePath $python -ArgumentList @($script) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru;Set-Content (Join-Path $root ($RunId+'.pid')) $process.Id;@{pid=$process.Id;started=(Get-Date -Format o);stdout=$stdout;stderr=$stderr;progress=(Join-Path $root 'progress.json')}|ConvertTo-Json|Set-Content (Join-Path $root ($RunId+'.meta.json'));Write-Output "started pid=$($process.Id) stdout=$stdout stderr=$stderr"






param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_visual_crop_stream_v1',[int]$DownloadPid=37632)

$ErrorActionPreference='Stop'
$root=Join-Path $RepoRoot 'artifacts\detached_nps_visual_crop_stream_score'
$logs=Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout=Join-Path $logs ($RunId+'_'+$timestamp+'.out.txt')
$stderr=Join-Path $logs ($RunId+'_'+$timestamp+'.err.txt')
$python='C:\Users\aaron\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe'
$output='D:\URAP_vatd_rank_results\nps_visual_crop_v1'
$arguments=@((Join-Path $RepoRoot 'tools\score_visual_crop_ranker_streaming.py'),'--model',(Join-Path $output 'model.pt'),'--predictionsgt-pkl','D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl','--frame-root','D:\URAP_nps_test_tvd\AllFrames\test','--out-score-map',(Join-Path $output 'visual_score_map.pkl'),'--progress-json',(Join-Path $root 'progress.json'),'--download-pid',[string]$DownloadPid,'--min-score','0.005','--batch-size','512')
$process=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content (Join-Path $root ($RunId+'.pid')) $process.Id
@{pid=$process.Id;started=(Get-Date -Format o);stdout=$stdout;stderr=$stderr;progress=(Join-Path $root 'progress.json');score_map=(Join-Path $output 'visual_score_map.pkl')}|ConvertTo-Json|Set-Content (Join-Path $root ($RunId+'.meta.json'))
Write-Output "started pid=$($process.Id) stdout=$stdout stderr=$stderr"

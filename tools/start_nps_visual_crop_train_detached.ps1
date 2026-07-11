param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_visual_crop_v1')

$ErrorActionPreference='Stop'
$root=Join-Path $RepoRoot 'artifacts\detached_nps_visual_crop_train'
$logs=Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout=Join-Path $logs ($RunId+'_'+$timestamp+'.out.txt')
$stderr=Join-Path $logs ($RunId+'_'+$timestamp+'.err.txt')
$python='C:\Users\aaron\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\python.exe'
$output='D:\URAP_vatd_rank_results\nps_visual_crop_v1'
New-Item -ItemType Directory -Force -Path $output | Out-Null
$arguments=@((Join-Path $RepoRoot 'tools\train_visual_crop_ranker.py'),'--predictionsgt-pkl','D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl','--frame-root','D:\URAP_nps_val_tvd\AllFrames\val','--out-model',(Join-Path $output 'model.pt'),'--out-summary',(Join-Path $output 'train_summary.json'),'--epochs','6','--batch-size','256','--image-size','128','--context-scale','4.0','--negative-min-score','0.005','--negative-ratio','6.0')
$process=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content (Join-Path $root ($RunId+'.pid')) $process.Id
@{pid=$process.Id;started=(Get-Date -Format o);stdout=$stdout;stderr=$stderr;model=(Join-Path $output 'model.pt');summary=(Join-Path $output 'train_summary.json')} | ConvertTo-Json | Set-Content (Join-Path $root ($RunId+'.meta.json'))
Write-Output "started pid=$($process.Id) stdout=$stdout stderr=$stderr"

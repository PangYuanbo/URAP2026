param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='modal_nps_visual_crop_score_v1')

$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:TERM='dumb'
$root=Join-Path $RepoRoot 'artifacts\detached_modal_nps_visual_crop_score'
$logs=Join-Path $root 'logs'
New-Item -ItemType Directory -Force -Path $logs | Out-Null
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout=Join-Path $logs ($RunId+'_'+$timestamp+'.out.txt')
$stderr=Join-Path $logs ($RunId+'_'+$timestamp+'.err.txt')
$process=Start-Process -FilePath 'C:\Users\aaron\.local\bin\modal.exe' -ArgumentList @('run',(Join-Path $RepoRoot 'tools\modal_score_nps_visual_crop.py')) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content (Join-Path $root ($RunId+'.pid')) $process.Id
@{pid=$process.Id;started=(Get-Date -Format o);stdout=$stdout;stderr=$stderr}|ConvertTo-Json|Set-Content (Join-Path $root ($RunId+'.meta.json'))
Write-Output "started pid=$($process.Id) stdout=$stdout stderr=$stderr"

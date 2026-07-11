param([string]$Destination = 'D:\URAP_nps_test_tvd')

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TERM = 'dumb'
$videos = Join-Path $Destination 'Videos'
$progress = Join-Path $Destination 'download_progress.json'
New-Item -ItemType Directory -Force -Path $videos | Out-Null
@{stage='download_videos';done=0;total=1;remote='/NPS/Videos/test';target=$videos;updated=(Get-Date -Format o)} | ConvertTo-Json | Set-Content $progress
& 'C:\Users\aaron\.local\bin\modal.exe' volume get urap-nps-formatted-v1 /NPS/Videos/test $videos --force
if ($LASTEXITCODE -ne 0) { throw 'NPS test video download failed' }
@{stage='done';done=1;total=1;remote='/NPS/Videos/test';target=$videos;updated=(Get-Date -Format o)} | ConvertTo-Json | Set-Content $progress

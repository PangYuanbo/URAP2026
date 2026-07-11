param([string]$Destination = 'D:\URAP_nps_test_tvd')

$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:TERM = 'dumb'
$frames = Join-Path $Destination 'AllFrames'
$progress = Join-Path $Destination 'frames_download_progress.json'
New-Item -ItemType Directory -Force -Path $frames | Out-Null
@{stage='download_frames';done=0;total=12355;remote='/NPS/AllFrames/test';target=$frames;updated=(Get-Date -Format o)} | ConvertTo-Json | Set-Content $progress
& 'C:\Users\aaron\.local\bin\modal.exe' volume get urap-nps-formatted-v1 /NPS/AllFrames/test $frames --force
if ($LASTEXITCODE -ne 0) { throw 'NPS test frame download failed' }
$count=(Get-ChildItem $frames -Recurse -File -Filter '*.png').Count
@{stage='done';done=$count;total=12355;remote='/NPS/AllFrames/test';target=$frames;updated=(Get-Date -Format o)} | ConvertTo-Json | Set-Content $progress

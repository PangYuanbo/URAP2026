param([string]$Destination='D:\URAP_nps_val_tvd')
$ErrorActionPreference='Stop';$env:PYTHONUTF8='1';$env:PYTHONIOENCODING='utf-8';$env:TERM='dumb'
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
$progress=Join-Path $Destination 'download_progress.json'
$items=@(
 @{volume='urap-nps-formatted-v1';remote='/NPS/AllFrames/val';local='AllFrames'},
 @{volume='urap-nps-formatted-v1';remote='/NPS/NPSvisdroneStyle/val/labels';local='NPSvisdroneStyle\val'},
 @{volume='urap-nps-formatted-v1';remote='/NPS/Videos/val';local='Videos'},
 @{volume='urap-model-weights-v1';remote='/TransVisDrone/NPS/best.pt';local='weights\best.pt'}
)
for($index=0;$index -lt $items.Count;$index++){$item=$items[$index];$target=Join-Path $Destination $item.local;if($item.remote -like '*.pt'){New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent)|Out-Null}else{New-Item -ItemType Directory -Force -Path $target|Out-Null};@{stage='download';done=$index;total=4;volume=$item.volume;remote=$item.remote;target=$target;updated=(Get-Date -Format o)}|ConvertTo-Json|Set-Content $progress;& 'C:\Users\aaron\.local\bin\modal.exe' volume get $item.volume $item.remote $target --force;if($LASTEXITCODE -ne 0){throw "download failed $($item.remote)"}}
@{stage='done';done=4;total=4;updated=(Get-Date -Format o)}|ConvertTo-Json|Set-Content $progress




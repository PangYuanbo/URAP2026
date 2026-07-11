param([string]$Destination='D:\URAP_nps_test_pack')

$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:TERM='dumb'
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
& 'C:\Users\aaron\.local\bin\modal.exe' volume get urap-nps-packs-v1 /NPS_AllFrames_test.tar $Destination --force
if($LASTEXITCODE -ne 0){throw 'archive download failed'}

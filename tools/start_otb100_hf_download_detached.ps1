$ErrorActionPreference='Stop'
$Repo='C:\Users\aaron\Desktop\URAP'
$Python=Join-Path $Repo 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$Run=Join-Path $Repo 'artifacts\detached_otb100_hf_download'
$Logs=Join-Path $Run 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$Stdout=Join-Path $Logs "otb100_hf_$Stamp.out.txt"
$Stderr=Join-Path $Logs "otb100_hf_$Stamp.err.txt"
$Process=Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList @(Join-Path $Repo 'tools\download_otb100_hf_archive.py') -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
@{pid=$Process.Id;start_time=(Get-Date).ToString('o');stdout=$Stdout;stderr=$Stderr;progress=(Join-Path $Run 'progress.json');target='D:\URAP_local_datasets\OTB100'} | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Run 'state.json')
$Process.Id | Set-Content -Encoding ASCII (Join-Path $Run 'pid.txt')
Write-Host "started pid=$($Process.Id) stdout=$Stdout stderr=$Stderr"

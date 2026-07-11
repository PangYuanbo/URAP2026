$ErrorActionPreference='Stop'
$repo='C:\Users\aaron\Desktop\URAP'
$runDir=Join-Path $repo 'artifacts\detached_tvd_official_last_val'
$logDir=Join-Path $runDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir|Out-Null
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$out=Join-Path $logDir "last_val_$timestamp.out.txt"
$err=Join-Path $logDir "last_val_$timestamp.err.txt"
$script=Join-Path $repo 'tools\run_tvd_nps_val.ps1'
$args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',$script,'-RepoDir','U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone','-DataYaml','U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone\data\NPS_URAP_D.yaml','-Project','U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone\runs\val\NPS_URAP','-Weights','D:\URAP_models\TransVisDrone_NPS_official\last.pt','-RunName','official_last_val_savegt','-BatchSize','8','-ExtraValArgs','--save-json-gt')
$process=Start-Process -FilePath 'powershell.exe' -ArgumentList $args -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err
$process.Id|Set-Content (Join-Path $runDir 'pid.txt') -Encoding ASCII
@{pid=$process.Id;start_time=(Get-Date).ToString('o');stdout_log=$out;stderr_log=$err;command=('powershell.exe '+($args -join ' '));expected_images=5944}|ConvertTo-Json|Set-Content (Join-Path $runDir 'meta.json') -Encoding UTF8
Write-Host "RUNNING PID: $($process.Id)"


param(
    [int]$WaitEvalPid = 0
)

$ErrorActionPreference = 'Stop'
$runDir = 'C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\runs\transvisdrone_ard100_overnight_pipeline'
$logDir = Join-Path $runDir 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$outLog = Join-Path $logDir "transvisdrone_ard100_overnight_pipeline_$ts.out.txt"
$errLog = Join-Path $logDir "transvisdrone_ard100_overnight_pipeline_$ts.err.txt"
$pidFile = Join-Path $runDir 'pid.txt'
$metaFile = Join-Path $runDir 'meta.json'
$script = 'C:\Users\aaron\Desktop\URAP\tools\run_transvisdrone_ard100_overnight_pipeline.ps1'

$args = @('-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',$script,'-WaitEvalPid',"$WaitEvalPid")
$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList $args -WorkingDirectory 'C:\Users\aaron\Desktop\URAP' -PassThru -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$proc.Id | Set-Content $pidFile -Encoding ASCII
@{
  pid = $proc.Id
  start_time = (Get-Date).ToString('s')
  wait_eval_pid = $WaitEvalPid
  stdout_log = $outLog
  stderr_log = $errLog
} | ConvertTo-Json | Set-Content $metaFile -Encoding UTF8

Write-Host 'RUNNING'
Write-Host "PID: $($proc.Id)"
Write-Host "stdout: $outLog"
Write-Host "stderr: $errLog"

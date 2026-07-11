param()
$ErrorActionPreference="Stop"
$repo="C:\Users\aaron\Desktop\URAP"; $runDir=Join-Path $repo "artifacts\detached_ard100_action_bank_v1"; $logDir=Join-Path $runDir "logs"; New-Item -ItemType Directory -Path $logDir -Force|Out-Null
$pidFile=Join-Path $runDir "pid.txt"; if(Test-Path $pidFile){$old=[int](Get-Content $pidFile -Raw);if(Get-Process -Id $old -ErrorAction SilentlyContinue){throw "ARD100 Action Bank job already running with PID $old"}}
$ts=Get-Date -Format "yyyyMMdd_HHmmss"; $outLog=Join-Path $logDir "action_bank_$ts.out.txt"; $errLog=Join-Path $logDir "action_bank_$ts.err.txt"
$py="C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$proc=Start-Process -FilePath $py -ArgumentList @("tools\run_ard100_action_bank_generalization.py") -WorkingDirectory $repo -PassThru -WindowStyle Hidden -RedirectStandardOutput $outLog -RedirectStandardError $errLog
$proc.Id|Set-Content $pidFile -Encoding ASCII; @{pid=$proc.Id;start_time=(Get-Date).ToString("o");stdout_log=$outLog;stderr_log=$errLog}|ConvertTo-Json|Set-Content (Join-Path $runDir "meta.json") -Encoding UTF8
Write-Host "RUNNING";Write-Host "PID: $($proc.Id)";Write-Host "stdout: $outLog";Write-Host "stderr: $errLog"

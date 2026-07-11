$ErrorActionPreference='Stop'
$Repo=Split-Path -Parent $PSScriptRoot
$Python=Join-Path $Repo 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$Script=Join-Path $Repo 'tools\run_tvd_samurai_memory_v93.py'
$Run=Join-Path $Repo 'artifacts\detached_tvd_samurai_memory_v93'
New-Item -ItemType Directory -Force -Path $Run|Out-Null
$PidFile=Join-Path $Run 'pid.txt';$Stdout=Join-Path $Run 'stdout.log';$Stderr=Join-Path $Run 'stderr.log';$Meta=Join-Path $Run 'start_meta.json'
if(Test-Path $PidFile){$OldPid=[int](Get-Content $PidFile -Raw).Trim();$Old=Get-CimInstance Win32_Process -Filter "ProcessId=$OldPid" -ErrorAction SilentlyContinue;if($Old){Write-Output "ALREADY RUNNING PID=$OldPid";exit 0}}
Set-Content $Stdout '' -Encoding utf8;Set-Content $Stderr '' -Encoding utf8;$Started=Get-Date
$Process=Start-Process -FilePath $Python -ArgumentList @($Script) -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -WindowStyle Hidden -PassThru
Set-Content $PidFile $Process.Id -Encoding ascii
@{pid=$Process.Id;start_time=$Started.ToString('o');command_line="$Python $Script";stdout=$Stdout;stderr=$Stderr}|ConvertTo-Json|Set-Content $Meta -Encoding utf8
Write-Output "STARTED PID=$($Process.Id) START=$($Started.ToString('o'))"

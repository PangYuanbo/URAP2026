param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='otb100_samurai_cmc_timebank_v2')
$ErrorActionPreference='Stop'
$root=Join-Path $RepoRoot 'artifacts\detached_otb100_samurai_cmc_timebank_v2'
$logs=Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logs|Out-Null
$active=@(Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like '*evaluate_samurai_otb100.py*' })
if($active.Count -gt 0){throw ('Conflicting SAMURAI GPU job active: '+(($active|ForEach-Object{$_.ProcessId}) -join ','))}
$python=Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
& $python (Join-Path $RepoRoot 'tools\download_otb100_github_releases.py') --verify-only
if($LASTEXITCODE -ne 0){throw 'OTB100 verification failed'}
$pidFile=Join-Path $root ($RunId+'.pid')
if(Test-Path $pidFile){$old=[int](Get-Content $pidFile|Select-Object -First 1);$oldProcess=Get-CimInstance Win32_Process -Filter "ProcessId=$old" -ErrorAction SilentlyContinue;if($oldProcess -and $oldProcess.CommandLine -like '*evaluate_samurai_otb100.py*'){throw "run already active pid=$old"}}
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout=Join-Path $logs ($RunId+'_'+$timestamp+'.out.txt')
$stderr=Join-Path $logs ($RunId+'_'+$timestamp+'.err.txt')
$script=Join-Path $RepoRoot 'tools\evaluate_samurai_otb100.py'
$progress=Join-Path $root 'progress.json'
$process=Start-Process -FilePath $python -ArgumentList @($script,'--progress-json',$progress) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content $pidFile $process.Id
@{pid=$process.Id;started=(Get-Date -Format o);command=@($python,$script,'--progress-json',$progress);stdout=$stdout;stderr=$stderr;progress=$progress;target='D:\URAP_vatd_rank_results\otb100_samurai_cmc_timebank_v2';total=100}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $root ($RunId+'.meta.json'))
Write-Output "started pid=$($process.Id) stdout=$stdout stderr=$stderr"

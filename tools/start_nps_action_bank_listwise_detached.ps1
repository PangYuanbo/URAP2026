param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='nps_action_bank_listwise_v3')
$ErrorActionPreference='Stop'
$root=Join-Path $RepoRoot 'artifacts\detached_nps_action_bank_listwise_v3';$logs=Join-Path $root 'logs';New-Item -ItemType Directory -Force $logs|Out-Null
$pidFile=Join-Path $root ($RunId+'.pid');if(Test-Path $pidFile){$old=[int](Get-Content $pidFile|Select-Object -First 1);if(Get-Process -Id $old -ErrorAction SilentlyContinue){throw "run already active pid=$old"}}
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss';$stdout=Join-Path $logs ($RunId+'_'+$timestamp+'.out.txt');$stderr=Join-Path $logs ($RunId+'_'+$timestamp+'.err.txt')
$python=Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe';$script=Join-Path $RepoRoot 'tools\run_nps_action_bank_listwise_official.py'
$process=Start-Process -FilePath $python -ArgumentList @($script) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content $pidFile $process.Id;@{pid=$process.Id;started=(Get-Date -Format o);command=@($python,$script);stdout=$stdout;stderr=$stderr;progress=(Join-Path $root 'progress.json')}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $root ($RunId+'.meta.json'))
Write-Output "started pid=$($process.Id) stdout=$stdout stderr=$stderr"

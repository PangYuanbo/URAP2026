param([string]$RepoRoot='C:\Users\aaron\Desktop\URAP',[string]$RunId='otb100_samurai_optflow_action_bank_v3')
$ErrorActionPreference='Stop'
$root=Join-Path $RepoRoot 'artifacts\detached_otb100_samurai_optflow_action_bank_v3'
$logs=Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logs|Out-Null
$python=Join-Path $RepoRoot 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$pidFile=Join-Path $root ($RunId+'.pid')
if(Test-Path $pidFile){$old=[int](Get-Content $pidFile|Select-Object -First 1);$proc=Get-CimInstance Win32_Process -Filter "ProcessId=$old" -ErrorAction SilentlyContinue;if($proc -and $proc.CommandLine -like '*postprocess_otb100_action_bank_cmc.py*'){throw "run already active pid=$old"}}
$timestamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout=Join-Path $logs ($RunId+'_'+$timestamp+'.out.txt')
$stderr=Join-Path $logs ($RunId+'_'+$timestamp+'.err.txt')
$script=Join-Path $RepoRoot 'tools\postprocess_otb100_action_bank_cmc.py'
$progress=Join-Path $root 'progress.json'
$output='D:\URAP_vatd_rank_results\otb100_samurai_optflow_action_bank_v3'
$process=Start-Process -FilePath $python -ArgumentList @($script,'--output-dir',$output,'--progress-json',$progress) -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content $pidFile $process.Id
@{pid=$process.Id;started=(Get-Date -Format o);command=@($python,$script,'--output-dir',$output,'--progress-json',$progress);stdout=$stdout;stderr=$stderr;progress=$progress;target=$output;total=100}|ConvertTo-Json -Depth 5|Set-Content (Join-Path $root ($RunId+'.meta.json'))
Write-Output "started pid=$($process.Id) stdout=$stdout stderr=$stderr"

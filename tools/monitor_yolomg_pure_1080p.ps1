param([Parameter(Mandatory=$true)][string]$RunId)
$repoRoot=Split-Path -Parent $PSScriptRoot
$runRoot=Join-Path $repoRoot ("artifacts\detached_yolomg_pure_1080p\"+$RunId)
$pidValue=[int](Get-Content (Join-Path $runRoot 'run.pid') -Raw)
$proc=Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue
if($proc -and $proc.CommandLine -like '*yolomg_pure_difference_1080p.py*'){Write-Output "RUNNING PID=$pidValue start=$((Get-Process -Id $pidValue).StartTime.ToString('o'))"}else{Write-Output "NOT RUNNING PID=$pidValue"}
$progress=Join-Path $runRoot 'progress.json';if(Test-Path $progress){Get-Content $progress -Raw}else{Write-Output 'progress=not-created'}
$stdout=Join-Path $runRoot 'stdout.log';$stderr=Join-Path $runRoot 'stderr.log';Write-Output "stdout=$stdout";Write-Output "stderr=$stderr";if(Test-Path $stdout){Get-Content $stdout -Tail 5};if((Test-Path $stderr)-and(Get-Item $stderr).Length-gt 0){Write-Output '--- stderr ---';Get-Content $stderr -Tail 12}
param(
    [Parameter(Mandatory=$true)][string]$RunId,
    [Parameter(Mandatory=$true)][string]$InputVideo
)
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "artifacts\venvs\nps_flow\Scripts\python.exe"
$script = Join-Path $repoRoot "tools\yolomg_pure_difference_1080p.py"
$runRoot = Join-Path $repoRoot ("artifacts\detached_yolomg_pure_1080p\" + $RunId)
$outputDir = Join-Path $repoRoot "artifacts\yolomg_pure_difference_1080p"
$stem = [IO.Path]::GetFileNameWithoutExtension($InputVideo)
$output = Join-Path $outputDir ($stem + "_yolomg_compensated_difference_1080p.mp4")
$pidPath = Join-Path $runRoot "run.pid"
$stdout = Join-Path $runRoot "stdout.log"
$stderr = Join-Path $runRoot "stderr.log"
$progress = Join-Path $runRoot "progress.json"
New-Item -ItemType Directory -Force -Path $runRoot,$outputDir | Out-Null
if(Test-Path $pidPath){$oldPid=[int](Get-Content $pidPath -Raw);$old=Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue;if($old -and $old.CommandLine -like '*yolomg_pure_difference_1080p.py*'){throw "Already running PID=$oldPid"}}
$arguments=@($script,'--input',$InputVideo,'--output',$output,'--progress-json',$progress)
$proc=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Set-Content $pidPath $proc.Id -Encoding ascii
Set-Content (Join-Path $runRoot 'started_at.txt') (Get-Date).ToString('o') -Encoding ascii
Write-Output "Started pure 1080p YOLOMG render."
Write-Output "PID=$($proc.Id)"
Write-Output "Output=$output"
Write-Output "Progress=$progress"
Write-Output "Logs=$stdout ; $stderr"
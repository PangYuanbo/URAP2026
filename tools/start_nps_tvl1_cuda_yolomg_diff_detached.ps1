param(
    [string]$RunId = "nps_tvl1_cuda_dji0619_sample",
    [string]$InputVideo = "C:\Users\aaron\Desktop\DJI_0619_W.MP4",
    [double]$DurationSeconds = 10,
    [int]$ProcessWidth = 960,
    [int]$DisplayWidth = 480
)
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot "artifacts\venvs\nps_flow_gpu\Scripts\python.exe"
$cudaBin = Join-Path $repoRoot "artifacts\opencv_cuda_tvl1_build\install\x64\vc17\bin"
$script = Join-Path $repoRoot "tools\nps_tvl1_cuda_yolomg_diff.py"
$runRoot = Join-Path $repoRoot ("artifacts\detached_nps_tvl1_cuda_yolomg\" + $RunId)
$outputDir = Join-Path $repoRoot ("artifacts\nps_tvl1_cuda_yolomg\" + $RunId)
$pidPath = Join-Path $runRoot "run.pid"; $stdout = Join-Path $runRoot "stdout.log"; $stderr = Join-Path $runRoot "stderr.log"; $progress = Join-Path $runRoot "progress.json"
New-Item -ItemType Directory -Force -Path $runRoot,$outputDir | Out-Null
if(Test-Path $pidPath){$oldPid=[int](Get-Content $pidPath -Raw);$old=Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue;if($old -and $old.CommandLine -like '*nps_tvl1_cuda_yolomg_diff.py*'){throw "Already running PID=$oldPid"}}
$arguments=@($script,'--input',$InputVideo,'--output-dir',$outputDir,'--duration-seconds',$DurationSeconds,'--process-width',$ProcessWidth,'--display-width',$DisplayWidth,'--progress-json',$progress)
$oldPath=$env:PATH
$env:PATH="$cudaBin;$oldPath"
try {$proc=Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru} finally {$env:PATH=$oldPath}
Set-Content $pidPath $proc.Id -Encoding ascii; Set-Content (Join-Path $runRoot 'started_at.txt') (Get-Date).ToString('o') -Encoding ascii
Write-Output "Started CUDA TV-L1 render PID=$($proc.Id)"; Write-Output "Output=$outputDir"; Write-Output "Logs=$stdout ; $stderr"
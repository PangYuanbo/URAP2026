param([string]$RunId = "opencv_cuda_tvl1")
$repoRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $repoRoot "artifacts\opencv_cuda_tvl1_build"
$worker = Join-Path $repoRoot "tools\build_opencv_cuda_tvl1_worker.ps1"
$pidPath = Join-Path $runRoot "build.pid"
$stdout = Join-Path $runRoot "stdout.log"
$stderr = Join-Path $runRoot "stderr.log"
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
if (Test-Path $pidPath) {
  $oldPid = [int](Get-Content $pidPath -Raw)
  $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
  if ($old -and $old.CommandLine -like "*build_opencv_cuda_tvl1_worker.ps1*") { throw "Build already running PID=$oldPid" }
}
$proc = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$worker,"-RepoRoot",$repoRoot,"-RunRoot",$runRoot) -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Set-Content -LiteralPath $pidPath -Value $proc.Id -Encoding ascii
Set-Content -LiteralPath (Join-Path $runRoot "started_at.txt") -Value (Get-Date).ToString("o") -Encoding ascii
Write-Output "Started CUDA OpenCV build PID=$($proc.Id)"
Write-Output "Logs=$stdout ; $stderr"
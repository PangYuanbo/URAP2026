param(
    [string]$RunDir = "C:\Users\aaron\Desktop\URAP\artifacts\joint_training\yolomg_nps_ard100_e20",
    [int]$IntervalSeconds = 7200
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$worker = Join-Path $PSScriptRoot "yolomg_checkpoint_snapshot_worker.ps1"
$runRoot = Join-Path $repoRoot "artifacts\joint_training\yolomg_snapshot_watcher"
$logRoot = Join-Path $runRoot "logs"
$pidPath = Join-Path $runRoot "snapshot.pid"
$metaPath = Join-Path $runRoot "snapshot.meta.json"
$trainingPidFile = Join-Path $repoRoot "artifacts\joint_training\yolomg_runner\train.pid"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
if (Test-Path -LiteralPath $pidPath) {
    $oldPid = Get-Content -LiteralPath $pidPath | Select-Object -First 1
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*yolomg_checkpoint_snapshot_worker.ps1*") { Write-Output "Already running PID=$oldPid"; exit 0 }
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logRoot "snapshot_$timestamp.out.txt"
$stderr = Join-Path $logRoot "snapshot_$timestamp.err.txt"
$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $worker, "-RunDir", $RunDir, "-TrainingPidFile", $trainingPidFile, "-IntervalSeconds", "$IntervalSeconds")
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ASCII
@{ pid = $process.Id; start_time = (Get-Date).ToString("o"); run_dir = $RunDir; interval_seconds = $IntervalSeconds; stdout = $stdout; stderr = $stderr; command = "powershell.exe $($arguments -join ' ')" } | ConvertTo-Json | Set-Content -LiteralPath $metaPath -Encoding UTF8
Write-Output "Started checkpoint snapshot watcher PID=$($process.Id) interval_seconds=$IntervalSeconds"

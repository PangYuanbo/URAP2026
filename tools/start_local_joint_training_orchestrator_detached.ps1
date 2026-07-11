param(
    [string]$DatasetRoot = "D:\URAP_local_datasets",
    [string]$NpsRoot = "",
    [string]$ArdRoot = "",
    [string]$ManifestRoot = "",
    [switch]$SkipDownloadWait,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$worker = Join-Path $PSScriptRoot "local_joint_training_orchestrator_worker.ps1"
$runRoot = Join-Path $repoRoot "artifacts\joint_training\orchestrator"
$logRoot = Join-Path $runRoot "logs"
$pidPath = Join-Path $runRoot "orchestrator.pid"
$metaPath = Join-Path $runRoot "orchestrator.meta.json"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
if (Test-Path -LiteralPath $pidPath) {
    $oldPid = Get-Content -LiteralPath $pidPath | Select-Object -First 1
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*local_joint_training_orchestrator_worker.ps1*") { Write-Output "Already running PID=$oldPid"; exit 0 }
    Write-Output "Previous orchestrator is NOT RUNNING. observed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') old_pid=$oldPid"
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logRoot "orchestrator_$timestamp.out.txt"
$stderr = Join-Path $logRoot "orchestrator_$timestamp.err.txt"
$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $worker, "-DatasetRoot", $DatasetRoot, "-PollSeconds", "$PollSeconds")
if ($NpsRoot) { $arguments += @("-NpsRoot", $NpsRoot) }
if ($ArdRoot) { $arguments += @("-ArdRoot", $ArdRoot) }
if ($ManifestRoot) { $arguments += @("-ManifestRoot", $ManifestRoot) }
if ($SkipDownloadWait) { $arguments += "-SkipDownloadWait" }
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ASCII
@{ pid = $process.Id; start_time = (Get-Date).ToString("o"); dataset_root = $DatasetRoot; nps_root = $NpsRoot; ard_root = $ArdRoot; manifest_root = $ManifestRoot; skip_download_wait = [bool]$SkipDownloadWait; stdout = $stdout; stderr = $stderr; command = "powershell.exe $($arguments -join ' ')" } | ConvertTo-Json | Set-Content -LiteralPath $metaPath -Encoding UTF8
Write-Output "Started detached local joint training orchestrator"
Write-Output "PID=$($process.Id)"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"

param(
    [string]$DestinationRoot = "D:\URAP_local_datasets",
    [switch]$IncludeAot
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$worker = Join-Path $PSScriptRoot "download_local_joint_datasets_worker.ps1"
$runRoot = Join-Path $repoRoot "artifacts\local_joint_dataset_download"
$logRoot = Join-Path $runRoot "logs"
$pidPath = Join-Path $runRoot "download.pid"
$metaPath = Join-Path $runRoot "download.meta.json"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = Get-Content -LiteralPath $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1
    $oldProcess = if ($oldPid) { Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue } else { $null }
    if ($oldProcess -and $oldProcess.CommandLine -like "*download_local_joint_datasets_worker.ps1*") {
        Write-Output "Already running PID=$oldPid"
        Get-Content -LiteralPath $metaPath -ErrorAction SilentlyContinue
        exit 0
    }
    Write-Output "Previous download is NOT RUNNING. observed=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') old_pid=$oldPid"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logRoot "download_$timestamp.out.txt"
$stderr = Join-Path $logRoot "download_$timestamp.err.txt"
$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $worker, "-DestinationRoot", $DestinationRoot)
if ($IncludeAot) { $arguments += "-IncludeAot" }
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ASCII
@{
    pid = $process.Id
    start_time = (Get-Date).ToString("o")
    destination = $DestinationRoot
    include_aot = [bool]$IncludeAot
    stdout = $stdout
    stderr = $stderr
    command = "powershell.exe $($arguments -join ' ')"
} | ConvertTo-Json | Set-Content -LiteralPath $metaPath -Encoding UTF8
Write-Output "Started detached local dataset download"
Write-Output "PID=$($process.Id)"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"

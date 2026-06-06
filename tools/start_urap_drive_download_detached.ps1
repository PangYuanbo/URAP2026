$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$statusDir = Join-Path $repoRoot "artifacts\urap_drive_download"
$worker = Join-Path $PSScriptRoot "download_urap_drive_files_worker.ps1"
$stdout = Join-Path $statusDir "download.stdout.log"
$stderr = Join-Path $statusDir "download.stderr.log"
$pidPath = Join-Path $statusDir "download.pid"
$startedPath = Join-Path $statusDir "download.started.txt"

New-Item -ItemType Directory -Force -Path $statusDir | Out-Null

$existingPid = $null
if (Test-Path -LiteralPath $pidPath) {
    $existingPid = (Get-Content -LiteralPath $pidPath -Raw).Trim()
}

if ($existingPid) {
    $existingProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existingProcess -and $existingProcess.CommandLine -like "*download_urap_drive_files_worker.ps1*") {
        Write-Output "Already running PID=$existingPid"
        Write-Output "stdout=$stdout"
        Write-Output "stderr=$stderr"
        exit 0
    }
}

$args = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$worker`"",
    "-RepoRoot", "`"$repoRoot`""
)

$process = Start-Process -FilePath "powershell.exe" `
    -ArgumentList $args `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

$process.Id | Set-Content -LiteralPath $pidPath -Encoding ASCII
(Get-Date).ToString("o") | Set-Content -LiteralPath $startedPath -Encoding ASCII

Write-Output "Started URAP Drive download"
Write-Output "PID=$($process.Id)"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"
Write-Output "pidFile=$pidPath"

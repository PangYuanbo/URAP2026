param(
    [string]$RunId = "yolomg_motion_process_drive_copy",
    [string]$Source = "C:\Users\aaron\Desktop\URAP\artifacts\yolomg_motion_process_site",
    [string]$Destination = "G:\My Drive\URAP2026\yolomg_motion_process_site_syncfix"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\aaron\Desktop\URAP"
$runRoot = Join-Path $repoRoot ("artifacts\detached_drive_copy\" + $RunId)
$logDir = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$sourceResolved = (Resolve-Path -LiteralPath $Source).Path
$destinationParent = Split-Path -Parent $Destination
New-Item -ItemType Directory -Force -Path $destinationParent | Out-Null

$stdoutLog = Join-Path $logDir ("runner_{0}.out.txt" -f $RunId)
$stderrLog = Join-Path $logDir ("runner_{0}.err.txt" -f $RunId)
$pidFile = Join-Path $runRoot "runner_pid.txt"
$metaFile = Join-Path $runRoot "runner_meta.txt"

$argString = '"{0}" "{1}" /E /Z /R:2 /W:5 /NFL /NDL /NP' -f $sourceResolved, $Destination

$startTime = Get-Date
$proc = Start-Process -FilePath "robocopy.exe" `
    -ArgumentList $argString `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Set-Content -Path $pidFile -Encoding ascii

@(
    "run_id=$RunId"
    "started=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
    "pid=$($proc.Id)"
    "source=$sourceResolved"
    "destination=$Destination"
    "stdout=$stdoutLog"
    "stderr=$stderrLog"
) | Set-Content -Path $metaFile -Encoding utf8

Write-Output "started=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Write-Output "pid=$($proc.Id)"
Write-Output "source=$sourceResolved"
Write-Output "destination=$Destination"
Write-Output "stdout=$stdoutLog"
Write-Output "stderr=$stderrLog"

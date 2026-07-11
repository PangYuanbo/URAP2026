param(
    [string]$RunId = "seqtrack_b384_weights",
    [string]$OutputRoot = "U:\URAP_models\seqtrack",
    [string]$Python = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$runRoot = Join-Path $repoRoot "artifacts\ata_reproduction\$RunId"
$logRoot = Join-Path $runRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot, $OutputRoot | Out-Null

$stdout = Join-Path $logRoot "download.out.txt"
$stderr = Join-Path $logRoot "download.err.txt"
$pidPath = Join-Path $runRoot "download.pid"
$metaPath = Join-Path $runRoot "download.meta.json"
$fileId = "1_OdMy1TKpTm3NgoW-FOObiz4_ROzCUvF"
$target = Join-Path $OutputRoot "train\seqtrack\seqtrack_b384\SEQTRACK_ep0500.pth.tar"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null

$arguments = @(
    (Join-Path $repoRoot "tools\download_gdrive_file.py"),
    "--file-id", $fileId,
    "--output", $target
)
$process = Start-Process -FilePath $Python -ArgumentList $arguments -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii

@{
    run_id = $RunId
    pid = $process.Id
    start_time = (Get-Date).ToString("o")
    command = "$Python $($arguments -join ' ')"
    target = $target
    stdout = $stdout
    stderr = $stderr
} | ConvertTo-Json | Set-Content -LiteralPath $metaPath -Encoding utf8

Write-Output "PID=$($process.Id)"
Write-Output "TARGET=$target"
Write-Output "STDOUT=$stdout"
Write-Output "STDERR=$stderr"

param(
    [string]$Annotations = "D:\datasets\my_video\heldout_annotation_workspace\annotations\qstr_real_boxes_heldout.csv",
    [string]$OutRoot = "D:\datasets\my_video\qstr_heldout_eval",
    [string]$Device = "0"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$script = Join-Path $PSScriptRoot "run_dji_heldout_eval.ps1"

if (-not (Test-Path $script)) {
    throw "Missing runner script: $script"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Join-Path $OutRoot ("logs\heldout_eval_" + $stamp)
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdout = Join-Path $logDir "stdout.log"
$stderr = Join-Path $logDir "stderr.log"
$pidFile = Join-Path $logDir "pid.txt"

$argsList = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $script,
    "-Annotations", $Annotations,
    "-OutRoot", $OutRoot,
    "-Device", $Device
)

$proc = Start-Process -FilePath "powershell.exe" `
    -ArgumentList $argsList `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Set-Content -Path $pidFile -Encoding ASCII

Write-Host "Started DJI held-out eval."
Write-Host "PID: $($proc.Id)"
Write-Host "Logs: $logDir"
Write-Host "Monitor with: powershell -ExecutionPolicy Bypass -File tools\monitor_dji_heldout_eval.ps1 -LogDir `"$logDir`""

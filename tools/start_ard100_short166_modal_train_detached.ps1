param(
    [ValidateSet("smoke", "train")]
    [string]$Mode = "smoke",
    [string]$RunName = "ard100_short166_modal_train_v1"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$logRoot = Join-Path $controlRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$pidPath = Join-Path $controlRoot "$Mode.pid"
$metaPath = Join-Path $controlRoot "$Mode.meta.json"
$stdoutPath = Join-Path $logRoot "$Mode.stdout.log"
$stderrPath = Join-Path $logRoot "$Mode.stderr.log"

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*modal_train_ard100_short166.py*") {
        throw "ARD100 short166 Modal $Mode is already running with PID $oldPid"
    }
}

$modal = Join-Path $env:USERPROFILE ".local\bin\modal.exe"
$arguments = @("run", "tools\modal_train_ard100_short166.py")
if ($Mode -eq "smoke") { $arguments += "--smoke-only" }
$previousUtf8 = $env:PYTHONUTF8
$env:PYTHONUTF8 = "1"
$process = Start-Process -FilePath $modal -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$env:PYTHONUTF8 = $previousUtf8

$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    mode = $Mode
    command = "$modal $($arguments -join ' ')"
    output_volume = "urap-ard100-samurai-short166-results-v1"
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
$metadata | ConvertTo-Json -Depth 4

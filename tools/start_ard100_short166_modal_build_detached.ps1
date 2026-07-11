param(
    [string]$RunName = "ard100_short166_modal_build_v1"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$logRoot = Join-Path $controlRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$pidPath = Join-Path $controlRoot "$RunName.pid"
$metaPath = Join-Path $controlRoot "$RunName.meta.json"
$stdoutPath = Join-Path $logRoot "$RunName.stdout.log"
$stderrPath = Join-Path $logRoot "$RunName.stderr.log"

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($old -and $old.CommandLine -like "*modal_build_ard100_short_tracklets.py*") {
        throw "ARD100 short166 Modal build is already running with PID $oldPid"
    }
}

$modal = Join-Path $env:USERPROFILE ".local\bin\modal.exe"
$arguments = @("run", "tools\modal_build_ard100_short_tracklets.py", "--split", "all")
$previousUtf8 = $env:PYTHONUTF8
$env:PYTHONUTF8 = "1"
$process = Start-Process -FilePath $modal -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$env:PYTHONUTF8 = $previousUtf8
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    command = "$modal $($arguments -join ' ')"
    expected_source_videos = 100
    expected_splits = 3
    output_volume = "urap-ard100-samurai-short166-v1"
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
$metadata | ConvertTo-Json -Depth 4

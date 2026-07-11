param([string]$RunName = "ard100_short166_modal_train_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$logRoot = Join-Path $controlRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$pidPath = Join-Path $controlRoot "sequence.pid"
$metaPath = Join-Path $controlRoot "sequence.meta.json"
$stdoutPath = Join-Path $logRoot "sequence.stdout.log"
$stderrPath = Join-Path $logRoot "sequence.stderr.log"

if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*sequence_ard100_short166_modal_train.ps1*") {
        throw "ARD100 short166 Modal train sequence is already running with PID $oldPid"
    }
}

$worker = Join-Path $PSScriptRoot "sequence_ard100_short166_modal_train.ps1"
$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $worker, "-RunName", $RunName)
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    command = "powershell.exe $($arguments -join ' ')"
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    state_file = (Join-Path $controlRoot "state.json")
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
$metadata | ConvertTo-Json -Depth 4

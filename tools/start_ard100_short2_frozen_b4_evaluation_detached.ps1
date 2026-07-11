param([string]$RunName = "ard100_short2_frozen_b4_evaluation")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
New-Item -ItemType Directory -Force -Path $controlRoot | Out-Null
$pidPath = Join-Path $controlRoot "$RunName.pid"
$metaPath = Join-Path $controlRoot "$RunName.meta.json"
$stdoutPath = Join-Path $controlRoot "$RunName.stdout.log"
$stderrPath = Join-Path $controlRoot "$RunName.stderr.log"
if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($oldProcess -and $oldProcess.CommandLine -like "*sequence_ard100_short2_frozen_b4_evaluation.ps1*") {
        throw "ARD100 short2 evaluation coordinator already runs with PID $oldPid"
    }
}
$script = Join-Path $PSScriptRoot "sequence_ard100_short2_frozen_b4_evaluation.ps1"
$process = Start-Process -FilePath "powershell.exe" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script) -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$metadata = [ordered]@{
    pid = $process.Id
    started_at = (Get-Date).ToString("o")
    command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File $script"
    state_file = (Join-Path $controlRoot "ard100_short2_frozen_b4_evaluation.state.json")
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
}
$metadata | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
$metadata | ConvertTo-Json -Depth 4

param([string]$RunName = "ard100_short166_experiment_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$logRoot = Join-Path $controlRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$pidPath = Join-Path $controlRoot "$RunName.pid"
$metaPath = Join-Path $controlRoot "$RunName.meta.json"
$stdout = Join-Path $logRoot "$RunName.stdout.log"
$stderr = Join-Path $logRoot "$RunName.stderr.log"
if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($old -and $old.CommandLine -like "*sequence_ard100_short166_experiment.ps1*") { throw "Coordinator already running: PID $oldPid" }
}
$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Join-Path $PSScriptRoot "sequence_ard100_short166_experiment.ps1"), "-RunName", $RunName)
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$meta = [ordered]@{ pid = $process.Id; started_at = (Get-Date).ToString("o"); command = "powershell.exe $($arguments -join ' ')"; stdout_log = $stdout; stderr_log = $stderr }
$meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
$meta | ConvertTo-Json -Depth 4

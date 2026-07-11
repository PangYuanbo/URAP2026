param([string]$RunName = "ard100_short166_local_build_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$logRoot = Join-Path $controlRoot "logs"
$pidPath = Join-Path $controlRoot "$RunName.pid"
$metaPath = Join-Path $controlRoot "$RunName.meta.json"
$stdout = Join-Path $logRoot "$RunName.stdout.log"
$stderr = Join-Path $logRoot "$RunName.stderr.log"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($old -and $old.CommandLine -like "*sequence_ard100_short166_local_build.ps1*") { throw "Local short166 build already running with PID $oldPid" }
}
$worker = Join-Path $PSScriptRoot "sequence_ard100_short166_local_build.ps1"
$arguments = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $worker, "-RunName", $RunName)
$process = Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$meta = [ordered]@{ pid = $process.Id; started_at = (Get-Date).ToString("o"); command = "powershell.exe $($arguments -join ' ')"; output_root = "U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI_SHORT166"; stdout_log = $stdout; stderr_log = $stderr }
$meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
$meta | ConvertTo-Json -Depth 4

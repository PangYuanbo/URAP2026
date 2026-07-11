param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$name = "wait_for_u_volume_idle"
$runner = Join-Path $repoRoot "tools\wait_for_urap_u_volume_idle.ps1"
$pidPath = Join-Path $controlRoot "$name.pid"
$metaPath = Join-Path $controlRoot "$name.meta.json"
$stdoutPath = Join-Path $controlRoot "$name.stdout.log"
$stderrPath = Join-Path $controlRoot "$name.stderr.log"
$progressPath = Join-Path $controlRoot "$name.progress.json"
New-Item -ItemType Directory -Force -Path $controlRoot | Out-Null
if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($old -and $old.CommandLine -like "*$runner*") { throw "Watcher already running with PID $oldPid" }
}
$args = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $runner, "-ProgressPath", $progressPath)
$proc = Start-Process -FilePath "powershell.exe" -ArgumentList $args -WorkingDirectory $repoRoot -WindowStyle Hidden     -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$proc.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
[ordered]@{pid=$proc.Id;started_at=(Get-Date).ToString("o");command="powershell.exe $($args -join ' ')";progress_file=$progressPath;stdout_log=$stdoutPath;stderr_log=$stderrPath} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
Get-Content -LiteralPath $metaPath -Raw

param()

$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
$name = "repair_urap_u_volume"
$metaPath = Join-Path $controlRoot "$name.meta.json"
if (-not (Test-Path -LiteralPath $metaPath)) {
    Write-Output "status=NOT STARTED"
    exit 0
}
$meta = Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json
$proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($meta.pid)" -ErrorAction SilentlyContinue
$running = [bool]($proc -and $proc.Name -eq "chkdsk.exe" -and $proc.CommandLine -like "*U:*")
$log = Get-Item -LiteralPath $meta.stdout_log -ErrorAction SilentlyContinue
$volume = Get-Volume -DriveLetter U -ErrorAction SilentlyContinue
Write-Output "status=$(if($running){'RUNNING'}else{'NOT RUNNING'}) pid=$($meta.pid) started=$($meta.started_at) last_output=$(if($log){$log.LastWriteTime.ToString('o')}else{'none'}) log=$($meta.stdout_log)"
if (Test-Path -LiteralPath $meta.stdout_log) { Get-Content -LiteralPath $meta.stdout_log -Tail 25 }
if (Test-Path -LiteralPath $meta.stderr_log) { Get-Content -LiteralPath $meta.stderr_log -Tail 10 }
if ($volume) { Write-Output "volume_health=$($volume.HealthStatus) operational=$($volume.OperationalStatus)" } else { Write-Output "volume=OFFLINE_OR_UNMOUNTED" }

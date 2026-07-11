param()

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs"
New-Item -ItemType Directory -Force -Path $controlRoot | Out-Null

$users = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.Name -notin @("powershell.exe", "pwsh.exe", "cmd.exe") -and $_.CommandLine -like "*U:\*"
})
if ($users.Count -gt 0) {
    $summary = ($users | ForEach-Object { "$($_.ProcessId):$($_.Name)" }) -join ", "
    throw "U: is still in use; refusing offline repair. Users: $summary"
}

$name = "repair_urap_u_volume"
$pidPath = Join-Path $controlRoot "$name.pid"
$metaPath = Join-Path $controlRoot "$name.meta.json"
$stdoutPath = Join-Path $controlRoot "$name.stdout.log"
$stderrPath = Join-Path $controlRoot "$name.stderr.log"
$progressPath = Join-Path $controlRoot "$name.progress.json"
if (Test-Path -LiteralPath $pidPath) {
    $oldPid = [int](Get-Content -LiteralPath $pidPath -Raw)
    $old = Get-CimInstance Win32_Process -Filter "ProcessId=$oldPid" -ErrorAction SilentlyContinue
    if ($old -and $old.CommandLine -like "*chkdsk*U:*") { throw "Repair already running with PID $oldPid" }
}

$arguments = @("U:", "/F", "/X")
$proc = Start-Process -FilePath "chkdsk.exe" -ArgumentList $arguments -WorkingDirectory $repoRoot -WindowStyle Hidden     -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
$proc.Id | Set-Content -LiteralPath $pidPath -Encoding ascii
$meta = [ordered]@{
    pid = $proc.Id
    started_at = (Get-Date).ToString("o")
    command = "chkdsk.exe $($arguments -join ' ')"
    stdout_log = $stdoutPath
    stderr_log = $stderrPath
    progress_file = $progressPath
}
$meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metaPath -Encoding utf8
[ordered]@{status="running";observed_at=(Get-Date).ToString("o");pid=$proc.Id;stdout_log=$stdoutPath;stderr_log=$stderrPath} | ConvertTo-Json | Set-Content -LiteralPath $progressPath -Encoding utf8
$meta | ConvertTo-Json -Depth 4

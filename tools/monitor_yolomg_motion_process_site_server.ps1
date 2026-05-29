param(
    [string]$RunId = "yolomg_motion_process_site_server"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\aaron\Desktop\URAP"
$runRoot = Join-Path $repoRoot ("artifacts\detached_motion_site_server\" + $RunId)
$pidFile = Join-Path $runRoot "runner_pid.txt"
$metaFile = Join-Path $runRoot "runner_meta.txt"

if (-not (Test-Path $pidFile)) {
    Write-Output "status=NOT RUNNING"
    Write-Output "done/total=0/0"
    Write-Output "pid="
    Write-Output "started="
    Write-Output "last_output_timestamp="
    Write-Output "last_completed_unit="
    Write-Output "stdout="
    Write-Output "stderr="
    exit 0
}

$meta = @{}
if (Test-Path $metaFile) {
    foreach ($line in Get-Content $metaFile) {
        if ($line -match "^(.*?)=(.*)$") {
            $meta[$matches[1]] = $matches[2]
        }
    }
}

$runPid = [int](Get-Content $pidFile -Raw).Trim()
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $runPid" -ErrorAction SilentlyContinue
$stdout = $meta["stdout"]
$stderr = $meta["stderr"]
$status = if ($null -ne $proc) { "RUNNING" } else { "NOT RUNNING" }
$lastOutTs = ""
if ($stdout -and (Test-Path $stdout)) {
    $lastOutTs = (Get-Item $stdout).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
}

Write-Output "status=$status"
Write-Output "done/total=1/1"
Write-Output "pid=$runPid"
Write-Output "started=$($meta["started"])"
Write-Output "last_output_timestamp=$lastOutTs"
Write-Output "last_completed_unit=server"
Write-Output "url=$($meta["url"])"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"

if ($stdout -and (Test-Path $stdout)) {
    Write-Output ""
    Write-Output "== stdout tail =="
    Get-Content $stdout -Tail 20
}

if ($stderr -and (Test-Path $stderr)) {
    Write-Output ""
    Write-Output "== stderr tail =="
    Get-Content $stderr -Tail 20
}

param(
    [string]$RunId = "yolomg_motion_process_drive_copy"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\aaron\Desktop\URAP"
$runRoot = Join-Path $repoRoot ("artifacts\detached_drive_copy\" + $RunId)
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
$source = $meta["source"]
$destination = $meta["destination"]
$stdout = $meta["stdout"]
$stderr = $meta["stderr"]
$status = if ($null -ne $proc) { "RUNNING" } else { "NOT RUNNING" }

$srcFiles = @()
$dstFiles = @()
if ($source -and (Test-Path $source)) {
    $srcFiles = Get-ChildItem -LiteralPath $source -Recurse -File -ErrorAction SilentlyContinue
}
if ($destination -and (Test-Path $destination)) {
    $dstFiles = Get-ChildItem -LiteralPath $destination -Recurse -File -ErrorAction SilentlyContinue
}
$srcCount = @($srcFiles).Count
$dstCount = @($dstFiles).Count
$srcBytes = ($srcFiles | Measure-Object -Property Length -Sum).Sum
$dstBytes = ($dstFiles | Measure-Object -Property Length -Sum).Sum
if ($null -eq $srcBytes) { $srcBytes = 0 }
if ($null -eq $dstBytes) { $dstBytes = 0 }

$lastOutTs = ""
if ($stdout -and (Test-Path $stdout)) {
    $lastOutTs = (Get-Item $stdout).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
}

Write-Output "status=$status"
Write-Output "done/total=$dstCount/$srcCount"
Write-Output "pid=$runPid"
Write-Output "started=$($meta["started"])"
Write-Output "last_output_timestamp=$lastOutTs"
Write-Output "last_completed_unit=$destination"
Write-Output "source=$source"
Write-Output "destination=$destination"
Write-Output ("source_bytes={0}" -f $srcBytes)
Write-Output ("destination_bytes={0}" -f $dstBytes)
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"

$driveProc = Get-Process -Name "GoogleDriveFS" -ErrorAction SilentlyContinue |
    Select-Object -First 3 Id, CPU, WorkingSet, StartTime, Path
if ($driveProc) {
    Write-Output ""
    Write-Output "== google drive desktop =="
    $driveProc
}

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

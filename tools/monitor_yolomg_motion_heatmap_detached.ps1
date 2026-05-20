param(
    [string]$RunId = "yolomg_motion_heatmap_02_05"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\aaron\Desktop\URAP"
$runRoot = Join-Path $repoRoot ("artifacts\detached_motion_heatmap\" + $RunId)
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
$outputDir = $meta["output_dir"]
$videos = @()
if ($meta.ContainsKey("videos")) {
    $videos = $meta["videos"] -split '\s+'
}
$total = $videos.Count
$done = 0
$lastCompleted = ""
foreach ($video in $videos) {
    if ([string]::IsNullOrWhiteSpace($video)) { continue }
    $manifest = Join-Path $outputDir ($video + "\manifest.txt")
    $heatmap = Join-Path $outputDir ($video + "\" + $video + "_motion_heatmap.mp4")
    if ((Test-Path $manifest) -and (Test-Path $heatmap)) {
        $done += 1
        $lastCompleted = $video
    }
}

$status = if ($null -ne $proc) { "RUNNING" } else { "NOT RUNNING" }
$lastOutTs = ""
if ($stdout -and (Test-Path $stdout)) {
    $lastOutTs = (Get-Item $stdout).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
}
$started = $meta["started"]

Write-Output "status=$status"
Write-Output "done/total=$done/$total"
Write-Output "pid=$runPid"
Write-Output "started=$started"
Write-Output "last_output_timestamp=$lastOutTs"
Write-Output "last_completed_unit=$lastCompleted"
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

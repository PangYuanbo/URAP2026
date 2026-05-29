param(
    [string]$RunId = "yolomg_motion_process_site_build"
)

$ErrorActionPreference = "Stop"

$repoRoot = "C:\Users\aaron\Desktop\URAP"
$runRoot = Join-Path $repoRoot ("artifacts\detached_motion_site\" + $RunId)
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
$webRoot = $meta["web_root"]
$stdoutText = ""
if ($stdout -and (Test-Path $stdout)) {
    $stdoutText = Get-Content $stdout -Raw -ErrorAction SilentlyContinue
    if ($null -eq $stdoutText) { $stdoutText = "" }
}

$matches = [regex]::Matches($stdoutText, "\[(DONE|SKIP)\]\s+(\d+)/(\d+)\s+(\S+)")
$done = 0
$total = 0
$lastCompleted = ""
if ($matches.Count -gt 0) {
    $m = $matches[$matches.Count - 1]
    $done = [int]$m.Groups[2].Value
    $total = [int]$m.Groups[3].Value
    $lastCompleted = $m.Groups[4].Value
}

$status = if ($null -ne $proc) { "RUNNING" } else { "NOT RUNNING" }
$lastOutTs = ""
if ($stdout -and (Test-Path $stdout)) {
    $lastOutTs = (Get-Item $stdout).LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
}

Write-Output "status=$status"
Write-Output "done/total=$done/$total"
Write-Output "pid=$runPid"
Write-Output "started=$($meta["started"])"
Write-Output "last_output_timestamp=$lastOutTs"
Write-Output "last_completed_unit=$lastCompleted"
Write-Output "stdout=$stdout"
Write-Output "stderr=$stderr"

try {
    $gpuLine = & nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>$null
    if ($gpuLine) {
        Write-Output "gpu=$gpuLine"
    }
} catch {
    Write-Output "gpu="
}

if ($webRoot -and (Test-Path $webRoot)) {
    $mp4Count = (Get-ChildItem -Path $webRoot -Recurse -File -Filter "*.mp4" -ErrorAction SilentlyContinue | Measure-Object).Count
    $posterCount = (Get-ChildItem -Path $webRoot -Recurse -File -Filter "*.jpg" -ErrorAction SilentlyContinue | Measure-Object).Count
    Write-Output "mp4_count=$mp4Count"
    Write-Output "poster_count=$posterCount"
    Write-Output "index=$webRoot\index.html"
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

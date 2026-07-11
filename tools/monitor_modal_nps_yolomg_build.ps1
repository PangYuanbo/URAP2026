param([string]$RepoRoot = "C:\Users\aaron\Desktop\URAP")

$env:PATH = "$HOME\.local\bin;$env:PATH"
$env:PYTHONUTF8 = "1"
$volume = "urap-nps-yolomg-v1"
$runnerDir = Join-Path $RepoRoot "artifacts\modal_nps_yolomg_build"
$pidPath = Join-Path $runnerDir "modal_nps_yolomg_build.pid"
$metaPath = Join-Path $runnerDir "modal_nps_yolomg_build.meta.txt"
$pidValue = if (Test-Path $pidPath) { (Get-Content $pidPath -Raw).Trim() } else { "" }
$process = if ($pidValue -match "^\d+$") { Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue } else { $null }
if ($process) { Write-Host "RUNNING=true PID=$pidValue"; Write-Host "PROCESS_COMMAND=$($process.CommandLine)" } else { Write-Host "NOT RUNNING PID=$pidValue" }
$globalDone = 0
$globalTotal = 0
foreach ($split in @("train", "val", "test")) {
    $temp = Join-Path $env:TEMP "urap_modal_nps_yolomg_progress_$split.json"
    Remove-Item $temp -Force -ErrorAction SilentlyContinue
    & modal volume get $volume "/NPS_YOLOMG/build_progress_$split.json" $temp 2>$null | Out-Null
    if (Test-Path $temp) {
        $progress = Get-Content $temp -Raw | ConvertFrom-Json
        $globalDone += [int]$progress.done
        $globalTotal += [int]$progress.total
        Write-Host "[$split] done/total=$($progress.done)/$($progress.total) clips=$($progress.clips_done)/$($progress.clips_total) last=$($progress.last_frame)"
    } else {
        Write-Host "[$split] done/total=0/? clips=0/$(@{train=36;val=4;test=10}[$split]) last=none"
    }
}
Write-Host "global done/total: $globalDone/$globalTotal"
if (Test-Path $metaPath) {
    $meta = @{}
    foreach ($line in Get-Content $metaPath) { if ($line -match "^([^=]+)=(.*)$") { $meta[$Matches[1]] = $Matches[2] } }
    Write-Host "start time: $($meta.started)"
    Write-Host "stdout log: $($meta.stdout)"
    Write-Host "stderr log: $($meta.stderr)"
    if ($meta.stdout -and (Test-Path $meta.stdout)) { Write-Host "== stdout tail =="; Get-Content $meta.stdout -Tail 12 }
    if ($meta.stderr -and (Test-Path $meta.stderr)) { Write-Host "== stderr tail =="; Get-Content $meta.stderr -Tail 12 }
}

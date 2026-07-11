param([string]$RepoRoot = "C:\Users\aaron\Desktop\URAP")

$env:PATH = "$HOME\.local\bin;$env:PATH"
$env:PYTHONUTF8 = "1"
$runnerDir = Join-Path $RepoRoot "artifacts\modal_nps_motion_validate"
$pidPath = Join-Path $runnerDir "modal_nps_motion_validate.pid"
$metaPath = Join-Path $runnerDir "modal_nps_motion_validate.meta.txt"
$pidValue = if (Test-Path $pidPath) { (Get-Content $pidPath -Raw).Trim() } else { "" }
$process = if ($pidValue -match "^\d+$") { Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue } else { $null }
if ($process) { Write-Host "RUNNING=true PID=$pidValue" } else { Write-Host "NOT RUNNING PID=$pidValue" }
foreach ($name in @("slow_0p5", "accelerate_g2", "decelerate_g2")) {
    $temp = Join-Path $env:TEMP "urap_integrity_$name.json"
    Remove-Item $temp -Force -ErrorAction SilentlyContinue
    & modal volume get urap-nps-motion-variants-v1 "/motion_v1/$name/integrity_progress.json" $temp 2>$null | Out-Null
    if (Test-Path $temp) {
        $progress = Get-Content $temp -Raw | ConvertFrom-Json
        Write-Host "[$name] done/total=$($progress.checked)/$($progress.total) last=$($progress.last_image)"
    } else { Write-Host "[$name] done/total=0/? last=none" }
}
if (Test-Path $metaPath) {
    $meta = @{}
    foreach ($line in Get-Content $metaPath) { if ($line -match "^([^=]+)=(.*)$") { $meta[$Matches[1]] = $Matches[2] } }
    Write-Host "start time: $($meta.started)"
    Write-Host "stdout log: $($meta.stdout)"
    Write-Host "stderr log: $($meta.stderr)"
    if ($meta.stdout -and (Test-Path $meta.stdout)) { Write-Host "== stdout tail =="; Get-Content $meta.stdout -Tail 12 }
    if ($meta.stderr -and (Test-Path $meta.stderr)) { Write-Host "== stderr tail =="; Get-Content $meta.stderr -Tail 12 }
}

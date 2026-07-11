param([string]$RepoRoot = "C:\Users\aaron\Desktop\URAP")
$env:PATH = "$HOME\.local\bin;$env:PATH"
$env:PYTHONUTF8 = "1"
$runner = Join-Path $RepoRoot "artifacts\modal_nps_motion_build"
$pidValue = if (Test-Path (Join-Path $runner "modal_nps_motion_build.pid")) { (Get-Content (Join-Path $runner "modal_nps_motion_build.pid") -Raw).Trim() } else { "" }
$process = if ($pidValue -match "^\d+$") { Get-CimInstance Win32_Process -Filter "ProcessId=$pidValue" -ErrorAction SilentlyContinue } else { $null }
if ($process) { Write-Host "RUNNING=true PID=$pidValue" } else { Write-Host "NOT RUNNING PID=$pidValue" }
$done = 0
foreach ($name in @("original", "slow_0p5", "fast_2x", "accelerate_g2", "decelerate_g2")) {
    $volume = if ($name -eq "original") { "urap-nps-motion-original-v1" } else { "urap-nps-motion-variants-v1" }
    $temp = Join-Path $env:TEMP "urap_motion_$name.json"
    Remove-Item $temp -Force -ErrorAction SilentlyContinue
    & modal volume get $volume "/motion_v1/progress_$name.json" $temp 2>$null | Out-Null
    if (Test-Path $temp) {
        $p = Get-Content $temp -Raw | ConvertFrom-Json
        $done += [int]$p.done
        Write-Host "[$name] done/total=$($p.done)/$($p.total) last=$($p.last_clip)"
    } else { Write-Host "[$name] done/total=0/10 last=none" }
}
Write-Host "global done/total: $done/50"
$metaPath = Join-Path $runner "meta.txt"
if (Test-Path $metaPath) {
    $meta = @{}; foreach ($line in Get-Content $metaPath) { if ($line -match "^([^=]+)=(.*)$") { $meta[$Matches[1]]=$Matches[2] } }
    Write-Host "stdout log: $($meta.stdout)"; Write-Host "stderr log: $($meta.stderr)"
    if ($meta.stdout -and (Test-Path $meta.stdout)) { Write-Host "== stdout tail =="; Get-Content $meta.stdout -Tail 10 }
    if ($meta.stderr -and (Test-Path $meta.stderr)) { Write-Host "== stderr tail =="; Get-Content $meta.stderr -Tail 10 }
}

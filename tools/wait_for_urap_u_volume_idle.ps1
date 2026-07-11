param(
    [string]$ProgressPath = "C:\Users\aaron\Desktop\URAP\artifacts\samurai_runs\wait_for_u_volume_idle.progress.json"
)

$ErrorActionPreference = "Stop"
function Write-State($payload) {
    $tmp = "$ProgressPath.$PID.tmp"
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $tmp -Encoding utf8
    Move-Item -LiteralPath $tmp -Destination $ProgressPath -Force
}

$idleObservations = 0
$requiredIdleObservations = 5
$eventLog = [IO.Path]::ChangeExtension($ProgressPath, ".events.log")
$lastSignature = $null
while ($true) {
    $users = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and $_.Name -notin @("powershell.exe", "pwsh.exe", "cmd.exe") -and $_.CommandLine -like "*U:\*"
    })
    $volume = Get-Volume -DriveLetter U -ErrorAction SilentlyContinue
    $rows = @($users | ForEach-Object {
        [ordered]@{pid=$_.ProcessId;parent_pid=$_.ParentProcessId;name=$_.Name;started_at=$_.CreationDate;command=$_.CommandLine}
    })
    $signature = ($rows | ForEach-Object { "$($_.pid):$($_.name)" }) -join ","
    if ($signature -ne $lastSignature) {
        "$(Get-Date -Format o) users=$($users.Count) signature=$signature" | Add-Content -LiteralPath $eventLog -Encoding utf8
        $lastSignature = $signature
    }
        if ($users.Count -eq 0) { $idleObservations++ } else { $idleObservations = 0 }
$status = if (-not $volume) { "volume_missing" } elseif ($users.Count) { "waiting_for_u_users" } elseif ($volume.HealthStatus -eq "Healthy" -and $volume.OperationalStatus -eq "OK") { "volume_healthy" } elseif ($idleObservations -ge $requiredIdleObservations) { "ready_for_offline_repair" } else { "confirming_u_idle" }
    Write-State ([ordered]@{
        status = $status
        observed_at = (Get-Date).ToString("o")
        user_count = $users.Count
        idle_observations = $idleObservations
        required_idle_observations = $requiredIdleObservations
        users = $rows
        volume_health = if($volume){[string]$volume.HealthStatus}else{$null}
        volume_operational_status = if($volume){[string]$volume.OperationalStatus}else{$null}
    })
    if ($status -in @("ready_for_offline_repair", "volume_healthy")) { exit 0 }
    Start-Sleep -Seconds 60
}

param([string]$DestinationRoot = "D:\URAP_local_datasets", [int]$TailLines = 20)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runRoot = Join-Path $repoRoot "artifacts\local_joint_dataset_download"
$pidPath = Join-Path $runRoot "download.pid"
$metaPath = Join-Path $runRoot "download.meta.json"
$stateRoot = Join-Path $DestinationRoot ".download_state"
$meta = if (Test-Path -LiteralPath $metaPath) { Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json } else { $null }
$pidValue = if (Test-Path -LiteralPath $pidPath) { Get-Content -LiteralPath $pidPath | Select-Object -First 1 } else { $null }
$process = if ($pidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
$validProcess = $process -and $process.CommandLine -like "*download_local_joint_datasets_worker.ps1*"

$plan = if (Test-Path -LiteralPath (Join-Path $stateRoot "plan.json")) { @(Get-Content -LiteralPath (Join-Path $stateRoot "plan.json") -Raw | ConvertFrom-Json) } else { @() }
$complete = @(Get-ChildItem -LiteralPath $stateRoot -Filter "*.complete.json" -File -ErrorAction SilentlyContinue | Where-Object Name -ne "all.complete.json")
$current = if (Test-Path -LiteralPath (Join-Path $stateRoot "current.json")) { Get-Content -LiteralPath (Join-Path $stateRoot "current.json") -Raw | ConvertFrom-Json } else { $null }
$latest = Get-ChildItem -LiteralPath $DestinationRoot -Recurse -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$downloadedBytes = (Get-ChildItem -LiteralPath $DestinationRoot -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
$driveName = ([IO.Path]::GetPathRoot($DestinationRoot)).Substring(0, 1)
$drive = Get-PSDrive -Name $driveName

if ($validProcess) {
    Write-Output "RUNNING"
    Write-Output "pid: $pidValue"
    Write-Output "start_time: $($meta.start_time)"
    Write-Output "command: $($process.CommandLine)"
} else {
    Write-Output "NOT RUNNING"
    Write-Output "last_pid: $pidValue"
    if ($meta) { Write-Output "start_time: $($meta.start_time)" }
}
Write-Output "done/total: $($complete.Count)/$($plan.Count)"
Write-Output "last_completed_unit: $(if($complete.Count){($complete | Sort-Object LastWriteTime | Select-Object -Last 1).BaseName}else{'none'})"
Write-Output "current_unit: $(if($current){$current.name}else{'none'})"
Write-Output "downloaded_gib: $([math]::Round($downloadedBytes / 1GB, 2))"
Write-Output "drive_free_gib: $([math]::Round($drive.Free / 1GB, 2))"
Write-Output "last_output_timestamp: $(if($latest){$latest.LastWriteTime.ToString('o')}else{'none'})"
Write-Output "last_output_file: $(if($latest){$latest.FullName}else{'none'})"
Write-Output "stdout: $(if($meta){$meta.stdout}else{'none'})"
Write-Output "stderr: $(if($meta){$meta.stderr}else{'none'})"
if ($meta -and (Test-Path -LiteralPath $meta.stdout)) { Write-Output "== stdout tail =="; Get-Content -LiteralPath $meta.stdout -Tail $TailLines }
if ($meta -and (Test-Path -LiteralPath $meta.stderr)) { Write-Output "== stderr tail =="; Get-Content -LiteralPath $meta.stderr -Tail $TailLines }

param(
  [string]$RunId = "nps_native_video_mvp_b32_w0_u_amp_ckpt",
  [string]$RunRoot = ""
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutDir = if ([string]::IsNullOrWhiteSpace($RunRoot)) { Join-Path $Repo "artifacts\native_video_detector\$RunId" } else { $RunRoot }
$PidFile = Join-Path $OutDir "train.pid"
$MetaFile = Join-Path $OutDir "train_meta.json"

if (-not (Test-Path $PidFile)) {
  throw "PID file not found: $PidFile"
}
if (-not (Test-Path $MetaFile)) {
  throw "Meta file not found: $MetaFile"
}

$Meta = Get-Content -LiteralPath $MetaFile -Raw | ConvertFrom-Json
$PidValue = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
$TrackedPids = @($PidValue)
try {
  $ChildPids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$PidValue" |
    Select-Object -ExpandProperty ProcessId
  if ($ChildPids) {
    $TrackedPids = @($TrackedPids + $ChildPids) | Select-Object -Unique
  }
} catch {
}

$StoppedAt = (Get-Date).ToString("o")
foreach ($TrackedPid in $TrackedPids) {
  Stop-Process -Id $TrackedPid -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

$Alive = @()
foreach ($TrackedPid in $TrackedPids) {
  if (Get-Process -Id $TrackedPid -ErrorAction SilentlyContinue) {
    $Alive += $TrackedPid
  }
}

$StopRecord = [ordered]@{
  run_id = $RunId
  stopped_at = $StoppedAt
  pid = $PidValue
  tracked_pids = $TrackedPids
  alive_after_stop = $Alive
  stdout_log = [string]$Meta.stdout_log
  stderr_log = [string]$Meta.stderr_log
}
$StopRecord | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $OutDir "stop_meta.json") -Encoding UTF8

Write-Output "RunId: $RunId"
Write-Output "StoppedAt: $StoppedAt"
Write-Output "PID: $PidValue"
Write-Output "TrackedPids: $($TrackedPids -join ', ')"
if ($Alive.Count -gt 0) {
  Write-Output "AliveAfterStop: $($Alive -join ', ')"
} else {
  Write-Output "AliveAfterStop: none"
}
Write-Output "StopMeta: $(Join-Path $OutDir 'stop_meta.json')"

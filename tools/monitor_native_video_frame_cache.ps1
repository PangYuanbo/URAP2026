param(
  [string]$RunId = "nps_native_video_frame_cache",
  [int]$Tail = 20
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutDir = Join-Path $Repo "artifacts\native_video_detector\$RunId"
$PidFile = Join-Path $OutDir "frame_cache.pid"
$MetaFile = Join-Path $OutDir "frame_cache_meta.json"

if (-not (Test-Path $MetaFile)) {
  throw "Meta file not found: $MetaFile"
}
if (-not (Test-Path $PidFile)) {
  throw "PID file not found: $PidFile"
}

$Meta = Get-Content -LiteralPath $MetaFile -Raw | ConvertFrom-Json
$PidValue = [int](Get-Content -LiteralPath $PidFile -Raw).Trim()
$Proc = Get-Process -Id $PidValue -ErrorAction SilentlyContinue
$ProcInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $PidValue" -ErrorAction SilentlyContinue
$Running = $null -ne $ProcInfo
$CommandLine = if ($ProcInfo) { [string]$ProcInfo.CommandLine } else { "" }
$CommandLineMatches = $Running -and ($CommandLine -like "*build_native_video_frame_cache.py*")
$StdoutLog = [string]$Meta.stdout_log
$StderrLog = [string]$Meta.stderr_log
$CacheDir = [string]$Meta.cache_dir
$Summary = Join-Path $CacheDir "cache_summary.json"
$LastProgress = $null
$DoneSummary = $null

if (Test-Path $StdoutLog) {
  foreach ($Line in (Get-Content -LiteralPath $StdoutLog -Tail 300 -ErrorAction SilentlyContinue)) {
    if ($Line -match '^\{') {
      try {
        $Obj = $Line | ConvertFrom-Json -ErrorAction Stop
        if ($Obj.kind -eq "native_video_frame_cache_progress") { $LastProgress = $Obj }
        if ($Obj.kind -eq "native_video_frame_cache_done") { $DoneSummary = $Obj }
      } catch {
      }
    }
  }
}
if ((-not $DoneSummary) -and (Test-Path $Summary)) {
  $DoneSummary = Get-Content -LiteralPath $Summary -Raw | ConvertFrom-Json
}

$Done = 0
$Total = [int]$Meta.frame_count
if ([int]$Meta.max_frames -gt 0 -and [int]$Meta.max_frames -lt $Total) {
  $Total = [int]$Meta.max_frames
}
if ($DoneSummary) {
  $Done = [int]$DoneSummary.total
} elseif ($LastProgress) {
  $Done = [int]$LastProgress.idx
}

$LastWrite = $null
foreach ($Path in @($StdoutLog, $StderrLog, $Summary)) {
  if ($Path -and (Test-Path $Path)) {
    $Time = (Get-Item -LiteralPath $Path).LastWriteTime
    if (-not $LastWrite -or $Time -gt $LastWrite) {
      $LastWrite = $Time
    }
  }
}

Write-Output "RunId: $RunId"
if ($Running) {
  Write-Output "Status: RUNNING"
  Write-Output "PID: $PidValue"
  Write-Output "ProcessStartTime: $($Proc.StartTime.ToString('o'))"
} else {
  Write-Output "Status: NOT RUNNING"
  Write-Output "PID: $PidValue"
}
Write-Output "command_line_matches: $CommandLineMatches"
Write-Output "command_line: $CommandLine"
Write-Output "done/total frames: $Done/$Total"
if ($LastProgress) {
  Write-Output "last completed unit: idx=$($LastProgress.idx) written=$($LastProgress.written) skipped=$($LastProgress.skipped) last=$($LastProgress.last)"
} elseif ($DoneSummary) {
  Write-Output "last completed unit: cache done written=$($DoneSummary.written) skipped=$($DoneSummary.skipped)"
} else {
  Write-Output "last completed unit: no progress JSON found yet"
}
Write-Output "last output timestamp: $LastWrite"
Write-Output "frames_dir: $($Meta.frames_dir)"
Write-Output "cache_dir: $CacheDir"
Write-Output "summary_json: $Summary"
Write-Output "stdout log: $StdoutLog"
Write-Output "stderr log: $StderrLog"

if (Test-Path $Summary) {
  Write-Output "--- cache summary ---"
  Get-Content -LiteralPath $Summary -Raw
}
if (Test-Path $StdoutLog) {
  Write-Output "--- stdout tail ---"
  Get-Content -LiteralPath $StdoutLog -Tail $Tail
}
if ((Test-Path $StderrLog) -and ((Get-Item -LiteralPath $StderrLog).Length -gt 0)) {
  Write-Output "--- stderr tail ---"
  Get-Content -LiteralPath $StderrLog -Tail $Tail
}

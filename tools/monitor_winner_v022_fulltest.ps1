param(
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_fulltest",
  [string]$RunId = "fulltest",
  [int]$Total = 172
)

$ErrorActionPreference = "Stop"

$runOut = Join-Path $OutputRoot $RunId
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path $metaFile) {
  Write-Host "== Meta =="
  $metaLines = Get-Content $metaFile | Select-Object -First 200
  $metaLines
  $stderrPath = ($metaLines | Where-Object { $_ -like 'stderr=*' } | Select-Object -First 1)
  if ($stderrPath) {
    $stderrPath = $stderrPath.Substring('stderr='.Length)
  } else {
    $stderrPath = $null
  }
}

if (-not (Test-Path $runOut)) {
  Write-Host "NO OUTPUT DIR: $runOut"
  exit 0
}

$dirs = Get-ChildItem $runOut -Directory -ErrorAction SilentlyContinue
$done = $dirs | Where-Object { Test-Path (Join-Path $_.FullName 'result.json') }
$latest = $done | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name,LastWriteTime

Write-Host "== Progress =="
Write-Host ("done={0}/{1} total_dirs={2}" -f $done.Count, $Total, $dirs.Count)
if ($latest) { $latest | Format-Table -AutoSize }

Write-Host "== Process =="
$procLine = "NOT RUNNING (no pid file)"
if (Test-Path $pidFile) {
  $runnerPid = (Get-Content $pidFile | Select-Object -First 1)
  if ($runnerPid -match '^\d+$') {
    $p = Get-Process -Id ([int]$runnerPid) -ErrorAction SilentlyContinue
    if ($null -ne $p) {
      $procLine = ("RUNNING pid={0} cpu={1} ws_mb={2:n1}" -f $p.Id, $p.CPU, ($p.WorkingSet64/1MB))
    } else {
      $procLine = ("NOT RUNNING (pid file exists but pid {0} not found)" -f $runnerPid)
    }
  } else {
    $procLine = "NOT RUNNING (pid file exists but cannot parse pid)"
  }
}
Write-Host $procLine

if ($stderrPath -and (Test-Path -Path $stderrPath -PathType Leaf)) {
  Write-Host "== Log Tail (stderr) =="
  try {
    $item = Get-Item $stderrPath
    Write-Host ("stderr_last_write={0}" -f $item.LastWriteTime)
    $tail = Get-Content -Tail 8 $stderrPath
    $last = ($tail | Where-Object { $_.Trim() } | Select-Object -Last 1)
    if ($last) { Write-Host $last }
  } catch {
    Write-Host ("failed_to_tail_stderr: {0}" -f $_.Exception.Message)
  }
}

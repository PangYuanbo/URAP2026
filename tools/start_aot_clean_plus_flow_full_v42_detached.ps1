param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$RunId = 'aot_clean_plus_flow_full_v42',
  [string]$OutputRoot = '',
  [int]$Workers = 4
)

$ErrorActionPreference = 'Stop'
if (-not $OutputRoot) {
  $OutputRoot = Join-Path $RepoRoot 'artifacts\route_b_official\aot_clean_plus_flow_full_v42_runner'
}
$python = Join-Path $RepoRoot 'papers\TransVisDrone\.venv\Scripts\python.exe'
$script = Join-Path $RepoRoot 'tools\aot_action_bank_flow_recovery_sharded.py'
$out = Join-Path $RepoRoot 'artifacts\route_b_official\aot_clean_plus_flow_full_v42'
$source = Join-Path $RepoRoot 'artifacts\route_b_official\aot_clean_full_v40_split_source\aotpredictions'
$tracklets = Join-Path $RepoRoot 'artifacts\route_b_official\aot_action_chunk_transfer_v1\tracklets_with_action_chunk_scores.jsonl'
$frames = 'U:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest\test'
$progress = Join-Path $OutputRoot "${RunId}_progress.json"
$logs = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $OutputRoot, $logs, $out | Out-Null
$pidFile = Join-Path $OutputRoot "${RunId}_pid.txt"
$metaFile = Join-Path $OutputRoot "${RunId}_meta.txt"
if (Test-Path -LiteralPath $pidFile) {
  $oldPid = Get-Content -LiteralPath $pidFile | Select-Object -First 1
  if ($oldPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*aot_action_bank_flow_recovery_sharded.py*') {
      Write-Host "Already running PID=$oldPid"
      Get-Content -LiteralPath $metaFile
      exit 0
    }
  }
}
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logs "${RunId}_${timestamp}.out.txt"
$stderr = Join-Path $logs "${RunId}_${timestamp}.err.txt"
$arguments = @(
  $script, '--repo-root', $RepoRoot, '--out-dir', $out, '--progress', $progress,
  '--source', $source, '--frames-root', $frames, '--tracklets', $tracklets,
  '--workers', [string]$Workers, '--appearance-search-fraction', '0'
)
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $RepoRoot -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$process.Id | Set-Content -LiteralPath $pidFile -Encoding ascii
@(
  "started=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
  "pid=$($process.Id)",
  "workers=$Workers",
  "source=$source",
  "output=$out",
  "progress=$progress",
  "stdout=$stdout",
  "stderr=$stderr"
) | Set-Content -LiteralPath $metaFile -Encoding utf8
Get-Content -LiteralPath $metaFile

param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
  [string]$LiRepoRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking'),
  [string]$UvExe = (Join-Path $env:USERPROFILE '.local\bin\uv.exe'),
  [string]$Checkpoint = '',
  [int[]]$Videos = @(41,42,43,44,45,46,47,48,49,50),
  [string]$OutRunRoot = '',
  [string]$OutGtCsv = '',
  [string]$OutSummary = '',
  [string]$OutDtDir = '',
  [string]$Profile = 'hard_recovery',
  [string]$DiagnosticsName = 'diagnostics_raw.jsonl',
  [double]$Score = 0.02,
  [double]$Nms = 0.5,
  [int]$MaxDetections = 300,
  [int]$MaxFrames = 0,
  [int]$FrameStride = 3,
  [int]$EmptyStride = 10,
  [int]$BatchSize = 1,
  [int]$NumWorkers = 0,
  [int]$MinSize = 1080,
  [int]$MaxSize = 1920,
  [string]$RunId = 'li_tetc_nps_proposal_export',
  [string]$OutputRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'artifacts\nps_li_tetc_compare\proposal_export_runner')
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -Path $RepoRoot -PathType Container)) { throw "RepoRoot not found: $RepoRoot" }
if (-not (Test-Path -Path $LiRepoRoot -PathType Container)) { throw "LiRepoRoot not found: $LiRepoRoot" }
if (-not (Test-Path -Path $UvExe -PathType Leaf)) { throw "uv not found: $UvExe" }
if (-not $Checkpoint) { $Checkpoint = Join-Path $LiRepoRoot 'pt_pipeline\runs\fasterrcnn_uav_v1.pt' }
if (-not (Test-Path -Path $Checkpoint -PathType Leaf)) { throw "Checkpoint not found: $Checkpoint" }

$defaultOutRoot = Join-Path $RepoRoot 'artifacts\nps_li_tetc_compare\li_frcnn_lowconf_proposals'
if (-not $OutRunRoot) { $OutRunRoot = Join-Path $defaultOutRoot 'run_root' }
if (-not $OutGtCsv) { $OutGtCsv = Join-Path $defaultOutRoot 'li_tetc_gt.csv' }
if (-not $OutSummary) { $OutSummary = Join-Path $defaultOutRoot 'export_summary.json' }
if (-not $OutDtDir) { $OutDtDir = Join-Path $defaultOutRoot 'dt' }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutSummary) | Out-Null
$logsDir = Join-Path $OutputRoot 'logs'
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pidFile = Join-Path $OutputRoot ("runner_{0}_pid.txt" -f $RunId)
$metaFile = Join-Path $OutputRoot ("runner_{0}_meta.txt" -f $RunId)

if (Test-Path -Path $pidFile -PathType Leaf) {
  $existingPid = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($existingPid -match '^\d+$') {
    $existing = Get-CimInstance Win32_Process -Filter "ProcessId = $existingPid" -ErrorAction SilentlyContinue
    if ($existing -and $existing.CommandLine -like '*export_li_tetc_proposals_for_action.py*') {
      Write-Host "Li-TETC NPS proposal export already running: pid=$existingPid"
      Get-Content $metaFile -ErrorAction SilentlyContinue
      exit 0
    }
  }
}

$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $logsDir ("runner_{0}_{1}.out.txt" -f $RunId, $ts)
$stderr = Join-Path $logsDir ("runner_{0}_{1}.err.txt" -f $RunId, $ts)
$script = Join-Path $RepoRoot 'tools\export_li_tetc_proposals_for_action.py'
$ptPipeline = Join-Path $LiRepoRoot 'pt_pipeline'

$argList = @(
  'run', 'python', $script,
  '--repo-root', $LiRepoRoot,
  '--ckpt', $Checkpoint,
  '--out-run-root', $OutRunRoot,
  '--profile', $Profile,
  '--diagnostics-name', $DiagnosticsName,
  '--out-gt-csv', $OutGtCsv,
  '--out-summary', $OutSummary,
  '--out-dt-dir', $OutDtDir,
  '--score', [string]$Score,
  '--nms', [string]$Nms,
  '--max-detections', [string]$MaxDetections,
  '--frame-stride', [string]$FrameStride,
  '--include-empty',
  '--empty-stride', [string]$EmptyStride,
  '--batch-size', [string]$BatchSize,
  '--num-workers', [string]$NumWorkers,
  '--min-size', [string]$MinSize,
  '--max-size', [string]$MaxSize,
  '--videos'
)
foreach ($video in $Videos) { $argList += [string]$video }
if ($MaxFrames -gt 0) { $argList += @('--max-frames', [string]$MaxFrames) }

$env:PYTHONPATH = "$RepoRoot;$ptPipeline"
$proc = Start-Process -FilePath $UvExe -ArgumentList $argList -WorkingDirectory $ptPipeline -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
$proc.Id | Set-Content -Encoding ascii -Path $pidFile
@(
  ('started={0}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')),
  ('pid={0}' -f $proc.Id),
  ('run_id={0}' -f $RunId),
  ('repo_root={0}' -f $RepoRoot),
  ('li_repo_root={0}' -f $LiRepoRoot),
  ('uv={0}' -f $UvExe),
  ('checkpoint={0}' -f $Checkpoint),
  ('videos={0}' -f ($Videos -join ',')),
  ('out_run_root={0}' -f $OutRunRoot),
  ('out_gt_csv={0}' -f $OutGtCsv),
  ('out_summary={0}' -f $OutSummary),
  ('out_dt_dir={0}' -f $OutDtDir),
  ('profile={0}' -f $Profile),
  ('diagnostics_name={0}' -f $DiagnosticsName),
  ('score={0}' -f $Score),
  ('max_frames={0}' -f $MaxFrames),
  ('frame_stride={0}' -f $FrameStride),
  ('empty_stride={0}' -f $EmptyStride),
  ('stdout={0}' -f $stdout),
  ('stderr={0}' -f $stderr),
  ('cmd_args={0}' -f ($argList -join ' '))
) | Set-Content -Encoding ascii -Path $metaFile

Write-Host 'Started detached Li-TETC NPS proposal export.'
Get-Content $metaFile

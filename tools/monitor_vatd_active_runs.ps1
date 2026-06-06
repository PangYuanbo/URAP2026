param(
  [int]$TailLines = 4
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$runs = @(
  @{
    name = 'aot_yolomg_shuffle_train'
    monitor = 'tools\monitor_route_b_vatd_motion_action_train.ps1'
    params = @{
      OutputRoot = 'artifacts\yolomg_action\vatd_motion_action_train_full_e1_b1024_crop64_nw0_nopin_detached'
      RunId = 'yolomg_train_vatd_motion_action_full_e1_b1024_crop64_nw0_nopin_20260605'
      TailLines = $TailLines
    }
  },
  @{
    name = 'nps_crop_full_train'
    monitor = 'tools\monitor_route_b_vatd_motion_action_train.ps1'
    params = @{
      OutputRoot = 'artifacts\nps_sota_research\tvd_nps_val_vatd_train_crop_full_runner_20260605'
      RunId = 'tvd_nps_val_vatd_train_crop_full_20260605'
      TailLines = $TailLines
    }
  },
  @{
    name = 'posttrain_orchestrator'
    monitor = 'tools\monitor_vatd_posttrain_orchestrator.ps1'
    params = @{ TailLines = $TailLines }
  },
  @{
    name = 'final_claim_summary_watcher'
    monitor = 'tools\monitor_vatd_claim_summary_watcher.ps1'
    params = @{
      OutputRoot = 'artifacts\vatd_claim_summary_final_runner_20260605'
      RunId = 'vatd_claim_summary_final_20260605'
      TailLines = $TailLines
    }
  }
)

foreach ($run in $runs) {
  Write-Host ""
  Write-Host "######## $($run.name) ########"
  $monitor = [string]$run['monitor']
  $paramMap = [hashtable]$run['params']
  & $monitor @paramMap
}

Write-Host ""
Write-Host "######## gpu_compute_process_details ########"
try {
  $gpuRows = & nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>$null
  if (-not $gpuRows) {
    Write-Host "gpu_compute_processes=none"
  } else {
    foreach ($row in $gpuRows) {
      $parts = $row -split ',', 3
      if ($parts.Count -lt 3) { continue }
      $gpuPid = [int]($parts[0].Trim())
      $processName = $parts[1].Trim()
      $usedMemoryMb = $parts[2].Trim()
      $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $gpuPid" -ErrorAction SilentlyContinue
      $commandLine = if ($proc) { [string]$proc.CommandLine } else { '' }
      $isRelevant = ($processName -match 'python') -or ($commandLine -match 'python|qstr|val\.py')
      if (-not $isRelevant) { continue }
      if ($proc) {
        Write-Host ("pid={0} ppid={1} gpu_mem_mb={2} process={3}" -f $gpuPid, $proc.ParentProcessId, $usedMemoryMb, $processName)
        Write-Host ("cmd={0}" -f $commandLine)
      } else {
        Write-Host ("pid={0} gpu_mem_mb={1} process={2} cmd=<not found>" -f $gpuPid, $usedMemoryMb, $processName)
      }
    }
  }
} catch {
  Write-Host ("gpu_compute_process_details_error={0}" -f $_.Exception.Message)
}

Write-Host ""
Write-Host "######## expected artifacts ########"
$artifacts = @(
  'artifacts\yolomg_action\vatd_motion_action_train_full_e1_b1024_crop64_nw0_nopin_20260605\vatd_motion_action.pt',
  'artifacts\nps_sota_research\tvd_nps_val_vatd_train_crop_full_20260605\vatd_motion_action.pt',
  'artifacts\route_b_official\aot_fulltest_vatd_motion_action_score_e1_shuffle_20260605\vatd_scores.jsonl',
  'artifacts\route_b_official\aot_fulltest_vatd_e1_shuffle_suppress_c0p10_b0p20_official_20260605\official_eval',
  'artifacts\nps_sota_research\tvd_nps_test_action_sweep_crop_full.json',
  'artifacts\route_b_official\aot_fulltest_vatd_e1_shuffle_suppress_c0p10_b0p20_official_20260605\aot_official_claim_comparison_claim_gate.json',
  'artifacts\nps_sota_research\tvd_nps_test_action_sweep_crop_full_comparison_claim_gate.json',
  'artifacts\vatd_claim_summary_final_20260605.json'
)

foreach ($path in $artifacts) {
  if (Test-Path $path) {
    $item = Get-Item $path
    if ($item.PSIsContainer) {
      Write-Host ("exists=true type=dir  path={0} last_write={1}" -f $path, $item.LastWriteTime)
    } else {
      Write-Host ("exists=true type=file path={0} length={1} last_write={2}" -f $path, $item.Length, $item.LastWriteTime)
    }
  } else {
    Write-Host ("exists=false path={0}" -f $path)
  }
}

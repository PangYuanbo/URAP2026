param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$OutputRoot = "C:\Users\aaron\Desktop\URAP\papers\TransVisDrone\runs\ablation\winner_port_v1",
  [string]$AOTRoot = "D:\URAP_datasets\AOT\part1",
  [string]$AOTOutRoot = "D:\URAP_datasets\TransVisDrone\AOT_part1_yolo_fulltest",
  [double]$AOTConfThres = 0.2,
  [int]$AOTBatchSize = 1,
  [int]$AOTImg = 1280,
  [int]$AOTNumFrames = 3,
  [int]$NPSBatchSize = 1,
  [int]$NPSImg = 1280,
  [int]$NPSNumFrames = 5,
  [double]$NPSConfThres = 0.001,
  [double]$NPSIouThres = 0.6
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Ensure-Dir([string]$d) {
  New-Item -ItemType Directory -Force -Path $d | Out-Null
}

function Write-State([hashtable]$state) {
  $statePath = Join-Path $OutputRoot "state.json"
  $state.timestamp = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
  ($state | ConvertTo-Json -Depth 8) | Set-Content -Encoding utf8 -Path $statePath
}

function Parse-NpsResults([string]$resultsTxtPath) {
  if (-not (Test-Path $resultsTxtPath)) { return $null }
  $lines = Get-Content $resultsTxtPath | Where-Object { $_.Trim() }
  if ($lines.Count -lt 2) { return $null }
  $metricLine = $lines[-1]
  $toks = ($metricLine -split '\s+') | Where-Object { $_ }
  # Expected: all, Images, Labels, P, R, mAP@.5, mAP@.5:.95, epoch
  if ($toks.Count -lt 8) { return $null }
  return @{
    p = [double]$toks[3]
    r = [double]$toks[4]
    map50 = [double]$toks[5]
    map = [double]$toks[6]
  }
}

function Parse-AotSummary([string]$summaryJsonPath) {
  if (-not (Test-Path $summaryJsonPath)) { return $null }
  $j = Get-Content $summaryJsonPath -Raw | ConvertFrom-Json
  return @{
    min_det_score = [double]$j.min_det_score
    fppi = [double]$j.fppi
    far = [double]$j.far
    det_dr_300_all = [double]$j.Detection.Encounters.'300'.All.dr
    track_dr_300_all = [double]$j.Tracking.Encounters.'300'.All.dr
  }
}

function Append-Row([hashtable]$row) {
  $csvPath = Join-Path $OutputRoot "results.csv"
  $exists = Test-Path $csvPath
  $obj = New-Object PSObject -Property $row
  if ($exists) {
    $obj | Export-Csv -NoTypeInformation -Append -Path $csvPath
  } else {
    $obj | Export-Csv -NoTypeInformation -Path $csvPath
  }
}

Ensure-Dir $OutputRoot
Ensure-Dir (Join-Path $OutputRoot "logs")

$resultsCsvPath = Join-Path $OutputRoot "results.csv"
$completed = @{}
if (Test-Path $resultsCsvPath) {
  try {
    Import-Csv $resultsCsvPath | ForEach-Object {
      if ($_.variant) { $completed[[string]$_.variant] = $true }
    }
  } catch {
    # If parsing fails, fall back to re-running everything (safer than skipping).
    Write-Host ("WARNING: failed_to_parse_existing_results_csv: {0}" -f $_.Exception.Message)
  }
}

$variants = @(
  @{ name = "baseline"; extra = @() },
  @{ name = "border10"; extra = @("--pp-border-margin", "10") },
  @{ name = "tracker"; extra = @("--pp-use-iou-tracker") },
  @{ name = "confirm"; extra = @(
      "--pp-use-iou-tracker",
      "--pp-track-confirm-len", "2",
      "--pp-track-confirm-only",
      "--pp-track-start-conf", "0.3",
      "--pp-track-continue-conf", "0.2"
    )
  }
)

$repoDir = Join-Path $URAPRoot "papers\\TransVisDrone"
$evalRoot = Join-Path $repoDir "runs\\eval\\AOT_URAP"
$npsProject = Join-Path $repoDir "runs\\val\\NPS_URAP"

Write-Host ("[{0}] TVD Winner-ablation runner starting" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
Write-Host ("OutputRoot: {0}" -f $OutputRoot)
Write-Host ("Variants: {0}" -f (($variants | ForEach-Object { $_.name }) -join ", "))

for ($i = 0; $i -lt $variants.Count; $i++) {
  $v = $variants[$i]
  $vname = [string]$v.name
  $extra = [string[]]$v.extra

  if ($completed.ContainsKey($vname)) {
    Write-Host ("[{0}] ({1}/{2}) skip completed variant: {3}" -f (Get-Date -Format "HH:mm:ss"), ($i+1), $variants.Count, $vname)
    continue
  }

  $tag = "wport_{0}" -f $vname

  Write-State @{ phase = "nps"; variant = $vname; index = $i; total = $variants.Count }
  Write-Host ("[{0}] ({1}/{2}) NPS val: {3}" -f (Get-Date -Format "HH:mm:ss"), ($i+1), $variants.Count, $vname)
  & (Join-Path $URAPRoot "tools\\run_tvd_nps_val.ps1") `
    -URAPRoot $URAPRoot `
    -RepoDir $repoDir `
    -Project $npsProject `
    -RunName $tag `
    -BatchSize $NPSBatchSize `
    -Img $NPSImg `
    -NumFrames $NPSNumFrames `
    -ConfThres $NPSConfThres `
    -IouThres $NPSIouThres `
    -ExtraValArgs $extra
  if ($LASTEXITCODE -ne 0) { throw "NPS val failed for variant $vname (exit $LASTEXITCODE)" }

  $npsResultsPath = Join-Path (Join-Path $npsProject $tag) "results.txt"
  $nps = Parse-NpsResults $npsResultsPath

  Write-State @{ phase = "aot"; variant = $vname; index = $i; total = $variants.Count }
  Write-Host ("[{0}] ({1}/{2}) AOT fulltest: {3}" -f (Get-Date -Format "HH:mm:ss"), ($i+1), $variants.Count, $vname)
  & (Join-Path $URAPRoot "tools\\run_aot_full_test.ps1") `
    -URAPRoot $URAPRoot `
    -AOTRoot $AOTRoot `
    -OutRoot $AOTOutRoot `
    -BatchSize $AOTBatchSize `
    -Img $AOTImg `
    -NumFrames $AOTNumFrames `
    -ConfThres $AOTConfThres `
    -RunNameSuffix $tag `
    -ExtraValArgs $extra `
    -SkipPrepare
  if ($LASTEXITCODE -ne 0) { throw "AOT fulltest failed for variant $vname (exit $LASTEXITCODE)" }

  # Find latest evaluation folder (evaluate_aot.py may increment the folder name).
  $runName = ("fulltest_conf{0}_{1}" -f ($AOTConfThres.ToString().Replace(".", "p")), $tag)
  $evalDir = Get-ChildItem -Directory -Path $evalRoot -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like ("{0}*" -f $runName) } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  $aot = $null
  $aotSummaryPath = $null
  if ($evalDir) {
    $sum = Get-ChildItem -Path (Join-Path $evalDir.FullName "summaries") -Filter "*.json" -ErrorAction SilentlyContinue |
      Sort-Object LastWriteTime -Descending |
      Select-Object -First 1
    if ($sum) {
      $aotSummaryPath = $sum.FullName
      $aot = Parse-AotSummary $sum.FullName
    }
  }

  $row = @{
    variant = $vname
    tag = $tag
    nps_results_txt = $npsResultsPath
    nps_p = if ($nps) { $nps.p } else { $null }
    nps_r = if ($nps) { $nps.r } else { $null }
    nps_map50 = if ($nps) { $nps.map50 } else { $null }
    nps_map = if ($nps) { $nps.map } else { $null }
    aot_summary_json = $aotSummaryPath
    aot_min_det_score = if ($aot) { $aot.min_det_score } else { $null }
    aot_fppi = if ($aot) { $aot.fppi } else { $null }
    aot_far = if ($aot) { $aot.far } else { $null }
    aot_det_dr_300_all = if ($aot) { $aot.det_dr_300_all } else { $null }
    aot_track_dr_300_all = if ($aot) { $aot.track_dr_300_all } else { $null }
    extra_args = ($extra -join " ")
    completed_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
  }
  Append-Row $row
  Write-Host ("[{0}] ({1}/{2}) variant done: {3}" -f (Get-Date -Format "HH:mm:ss"), ($i+1), $variants.Count, $vname)
}

Write-State @{ phase = "done"; variant = ""; index = $variants.Count; total = $variants.Count }
Write-Host ("[{0}] All variants done." -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))

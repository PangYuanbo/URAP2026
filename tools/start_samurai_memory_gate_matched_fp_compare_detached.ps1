param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$Python = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe",
  [string]$ImagesList = "U:\URAP_datasets\ARD100_YOLOMG\val.txt",
  [string]$BaselineLabelDir = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\yolomg_val_full_no_ncc_eval\pred_labels",
  [string]$SamuraiLabelDir = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\samurai_memory_gate_yolomg_val_full_eval\pred_labels",
  [string]$BaselineMethod = "no_ncc",
  [string]$SamuraiMethod = "samurai_gate",
  [double]$BaselineThreshold = 0.001,
  [string]$Thresholds = "0.001 0.01 0.05 0.1 0.2 0.3 0.5 0.7 0.9",
  [int]$ImageWidth = 1920,
  [int]$ImageHeight = 1080,
  [int]$MaxFrames = 0,
  [int]$ProgressEvery = 5000,
  [string]$OutRoot = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\matched_fp_no_ncc_vs_samurai_gate",
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\matched_fp_no_ncc_vs_samurai_gate_runner",
  [string]$RunId = "matched_fp_no_ncc_vs_samurai_gate"
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -Path $BaselineLabelDir -PathType Container)) { throw "Missing baseline label dir: $BaselineLabelDir" }
if (-not (Test-Path -Path $SamuraiLabelDir -PathType Container)) { throw "Missing SAMURAI gate label dir: $SamuraiLabelDir" }

& (Join-Path $PSScriptRoot "start_compare_yolo_labels_matched_fp_detached.ps1") `
  -URAPRoot $URAPRoot `
  -Python $Python `
  -ImagesList $ImagesList `
  -Method @("$BaselineMethod=$BaselineLabelDir", "$SamuraiMethod=$SamuraiLabelDir") `
  -BaselineMethod $BaselineMethod `
  -BaselineThreshold $BaselineThreshold `
  -Thresholds $Thresholds `
  -ImageWidth $ImageWidth `
  -ImageHeight $ImageHeight `
  -MaxFrames $MaxFrames `
  -ProgressEvery $ProgressEvery `
  -OutRoot $OutRoot `
  -RunRoot $RunRoot `
  -RunId $RunId

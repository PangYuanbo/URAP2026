param(
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\matched_fp_no_ncc_vs_samurai_gate_runner",
  [string]$RunId = "matched_fp_no_ncc_vs_samurai_gate",
  [int]$TailLines = 40
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "monitor_compare_yolo_labels_matched_fp.ps1") -RunRoot $RunRoot -RunId $RunId -TailLines $TailLines

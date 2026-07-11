param(
  [string]$URAPRoot = "C:\Users\aaron\Desktop\URAP",
  [string]$Python = "C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe",
  [string]$ImagesList = "U:\URAP_datasets\ARD100_YOLOMG\val.txt",
  [string]$TrajectoryCsv = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\samurai_memory_gate_yolomg_val_full\trajectory.csv",
  [string]$OutRoot = "U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\samurai_memory_gate_yolomg_val_full_eval",
  [string]$RunRoot = "C:\Users\aaron\Desktop\URAP\artifacts\detector_first_temporal_recovery\samurai_memory_gate_yolomg_val_eval_runner",
  [string]$RunId = "samurai_memory_gate_yolomg_val_eval",
  [double]$ConfThres = 0.001,
  [double]$MatchIou = 0.5
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "start_temporal_recovery_yolomg_val_eval_detached.ps1") `
  -URAPRoot $URAPRoot `
  -Python $Python `
  -ImagesList $ImagesList `
  -TrajectoryCsv $TrajectoryCsv `
  -OutRoot $OutRoot `
  -RunRoot $RunRoot `
  -RunId $RunId `
  -ConfThres $ConfThres `
  -MatchIou $MatchIou

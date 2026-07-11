param([string]$RunName = "finetune_base_plus_ard100_short166_stage1")

$ErrorActionPreference = "Stop"
$preflight = & (Join-Path $PSScriptRoot "check_ard100_short166_local_ready.ps1")
if ($LASTEXITCODE -ne 0) { throw "ARD100 short166 local training preflight failed" }
& (Join-Path $PSScriptRoot "start_samurai_finetune_detached.ps1") -Config "configs/sam2.1_training/sam2.1_hiera_b+_ARD100_short166_local_stage1.yaml" -RunName $RunName -TotalEpochs 16 -RunRoot "U:\URAP_runs\samurai\finetune_base_plus_ard100_short166_stage1"

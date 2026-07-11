$ErrorActionPreference = 'Continue'
Write-Output '=== V113 DENSE CANDIDATES ==='
& 'C:\Users\aaron\Desktop\URAP\tools\monitor_tvd_train_dense_candidates_v113.ps1' | Select-Object -First 12
Write-Output '=== V114 LABEL/DISTRIBUTION CHECK ==='
& 'C:\Users\aaron\Desktop\URAP\tools\monitor_tvd_dense_postcheck_v114.ps1' | Select-Object -First 10
Write-Output '=== V115 ACTION FEATURES ==='
& 'C:\Users\aaron\Desktop\URAP\tools\monitor_tvd_dense_action_features_v115.ps1' | Select-Object -First 9
Write-Output '=== V116 MODEL/EVALUATION ==='
& 'C:\Users\aaron\Desktop\URAP\tools\monitor_tvd_dense_action_model_v116.ps1' | Select-Object -First 9
Write-Output '=== V117 TEMPORAL EXPERT ==='
& 'C:\Users\aaron\Desktop\URAP\tools\monitor_tvd_dense_temporal_gate_v117.ps1' | Select-Object -First 9
Write-Output '=== V119 DOMAIN-BALANCED ==='
& 'C:\Users\aaron\Desktop\URAP\tools\monitor_tvd_domain_balanced_action_v119.ps1' | Select-Object -First 9
Write-Output '=== V120 OOF STACK ==='
& 'C:\Users\aaron\Desktop\URAP\tools\monitor_tvd_oof_stack_v120.ps1' | Select-Object -First 9
Write-Output '=== V118 FINAL COMPARISON ==='
& 'C:\Users\aaron\Desktop\URAP\tools\monitor_ard100_tvd_final_compare_v118.ps1' | Select-Object -First 9
Write-Output '=== GPU ==='
& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader





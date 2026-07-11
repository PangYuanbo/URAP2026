$ErrorActionPreference = 'Stop'
$Repo = 'C:\Users\aaron\Desktop\URAP'
$Python = Join-Path $Repo 'URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe'
$Run = Join-Path $Repo 'artifacts\detached_nps_action_bank_correction_v6'
$Logs = Join-Path $Run 'logs'
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$Stdout = Join-Path $Logs "nps_action_bank_correction_v6_$Stamp.out.txt"
$Stderr = Join-Path $Logs "nps_action_bank_correction_v6_$Stamp.err.txt"
$Out = 'D:\URAP_vatd_rank_results\nps_action_bank_correction_v6'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Arguments = @(
  (Join-Path $Repo 'tools\train_action_bank_correction_head.py'),
  '--train-pkl', 'D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0.pkl',
  '--train-aux', 'D:\URAP_vatd_rank_results\nps_action_bank_listwise_v3\train_tracklets_action_bank.jsonl',
  '--val-pkl', 'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl',
  '--val-aux', 'D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\val_tracklets_action_bank.jsonl',
  '--test-pkl', 'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl',
  '--test-aux', 'D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\test_tracklets_action_bank.jsonl',
  '--out-model', (Join-Path $Out 'model.pt'),
  '--out-val-scores', (Join-Path $Out 'val_scores.jsonl'),
  '--out-test-scores', (Join-Path $Out 'test_scores.jsonl'),
  '--out-summary', (Join-Path $Out 'train_summary.json'),
  '--epochs', '18', '--batch-size', '4096', '--hidden', '192', '--lr', '0.0005', '--device', 'cuda'
)
$Process = Start-Process -WindowStyle Hidden -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Repo -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
$State = [ordered]@{ pid = $Process.Id; start_time = (Get-Date).ToString('o'); stdout = $Stdout; stderr = $Stderr; output = $Out; epochs = 18 }
$State | ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $Run 'state.json')
$Process.Id | Set-Content -Encoding ASCII (Join-Path $Run 'pid.txt')
$State | ConvertTo-Json

$ErrorActionPreference = 'Stop'
$MainPidFile = 'C:\Users\aaron\Desktop\URAP\artifacts\detached_tvd_train_dense_candidates_v113\pid.txt'
$Python = 'U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone\.venv\Scripts\python.exe'
$Repo = 'C:\Users\aaron\Desktop\URAP'
$TvdRoot = 'U:\URAP_cold_storage\Desktop_URAP\papers\TransVisDrone'
$Out = 'D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\postcheck_v114'
$DenseRaw = 'D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\official_train_dense\predictionsgt\predictionsgt_split_0.pkl'
$Dense = 'D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\official_train_dense\predictionsgt\predictionsgt_split_0_fixed_canvas.pkl'
$Old = 'D:\URAP_nps_train_tvd\runs\nps_train_rank_source\predictionsgt\predictionsgt_split_0_fixed_canvas.pkl'
$Val = 'D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl'
$Test = 'D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl'
New-Item -ItemType Directory -Force -Path $Out | Out-Null
while ($true) {
    $MainPid = if (Test-Path -LiteralPath $MainPidFile) { [int](Get-Content -LiteralPath $MainPidFile -Raw).Trim() } else { 0 }
    $Main = if ($MainPid) { Get-CimInstance Win32_Process -Filter "ProcessId=$MainPid" -ErrorAction SilentlyContinue } else { $null }
    if (-not $Main) { break }
    Write-Output "WAITING main_pid=$MainPid time=$((Get-Date).ToString('o'))"
    Start-Sleep -Seconds 30
}
if (-not (Test-Path -LiteralPath $DenseRaw)) { throw "Dense prediction output missing after main process stopped: $DenseRaw" }
Write-Output "LABEL_FIX_START $((Get-Date).ToString('o'))"
& $Python "$Repo\tools\fix_nps_train_predictionsgt_canvas.py" --source $DenseRaw --output $Dense --summary "$Out\label_canvas_fix.json"
if ($LASTEXITCODE -ne 0) { throw "label canvas fixer failed: $LASTEXITCODE" }
Write-Output "ANALYZE_START $((Get-Date).ToString('o'))"
& $Python "$Repo\tools\analyze_tvd_dense_candidates_v114.py" --source dense_train $Dense --source old_train $Old --source val $Val --source test $Test --out-json "$Out\density_alignment_summary.json"
if ($LASTEXITCODE -ne 0) { throw "density analyzer failed: $LASTEXITCODE" }
& $Python "$Repo\tools\eval_tvd_predictionsgt_pkl.py" --tvd-root $TvdRoot --predictionsgt-pkl $Dense --out-json "$Out\dense_train_detector_baseline.json"
if ($LASTEXITCODE -ne 0) { throw "detector evaluator failed: $LASTEXITCODE" }
Write-Output "POSTCHECK_DONE $((Get-Date).ToString('o'))"

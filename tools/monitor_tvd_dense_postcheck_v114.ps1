$ErrorActionPreference = 'Stop'
$Run = 'C:\Users\aaron\Desktop\URAP\artifacts\detached_tvd_dense_postcheck_v114'
$WorkerPid = [int](Get-Content -LiteralPath (Join-Path $Run 'pid.txt') -Raw).Trim()
$Process = Get-CimInstance Win32_Process -Filter "ProcessId=$WorkerPid" -ErrorAction SilentlyContinue
$Meta = Get-Content -LiteralPath (Join-Path $Run 'start_meta.json') -Raw | ConvertFrom-Json
$LabelFix = 'D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\postcheck_v114\label_canvas_fix.json'
$Density = 'D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\postcheck_v114\density_alignment_summary.json'
$Baseline = 'D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\postcheck_v114\dense_train_detector_baseline.json'
$Done = [int](Test-Path -LiteralPath $LabelFix) + [int](Test-Path -LiteralPath $Density) + [int](Test-Path -LiteralPath $Baseline)
$Stage = if ($Done -eq 3) { 'done' } elseif ($Done -gt 0) { 'analyzing' } else { 'waiting_main' }
$Logs = @($Meta.stdout, $Meta.stderr) | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object { Get-Item -LiteralPath $_ }
$Last = $Logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Output "STATUS=$(if($Process){'RUNNING'}else{'NOT RUNNING'})"
Write-Output "DONE_TOTAL=$Done/3 STAGE=$Stage"
Write-Output "PID=$WorkerPid START=$($Meta.start_time)"
if ($Process) { Write-Output "COMMAND=$($Process.CommandLine)" }
if ($Last) { Write-Output "LAST_OUTPUT=$($Last.LastWriteTime.ToString('o')) FILE=$($Last.FullName)" }
Write-Output "STDOUT=$($Meta.stdout)"
Write-Output "STDERR=$($Meta.stderr)"
Get-Content -LiteralPath $Meta.stdout -Tail 8 -ErrorAction SilentlyContinue
Get-Content -LiteralPath $Meta.stderr -Tail 8 -ErrorAction SilentlyContinue

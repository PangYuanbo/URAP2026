$ErrorActionPreference = 'Stop'
$Run = 'C:\Users\aaron\Desktop\URAP\artifacts\detached_tvd_train_dense_candidates_v113'
$PidFile = Join-Path $Run 'pid.txt'; $MetaFile = Join-Path $Run 'start_meta.json'; $Stdout = Join-Path $Run 'stdout_batch16.log'; $Stderr = Join-Path $Run 'stderr_batch16.log'
$OutputDir = 'D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\official_train_dense'; $PredictionFile = Join-Path $OutputDir 'predictionsgt\predictionsgt_split_0.pkl'; $TotalFrames = 51951
if (-not (Test-Path -LiteralPath $PidFile)) { Write-Output "NOT RUNNING: PID file missing: $PidFile"; exit 1 }
$PidValue = [int](Get-Content -LiteralPath $PidFile -Raw).Trim(); $Process = Get-CimInstance Win32_Process -Filter "ProcessId=$PidValue" -ErrorAction SilentlyContinue
$Meta = if (Test-Path -LiteralPath $MetaFile) { Get-Content -LiteralPath $MetaFile -Raw | ConvertFrom-Json } else { $null }; $Status = if ($Process) { 'RUNNING' } else { 'NOT RUNNING' }
$CombinedTail = @(); foreach ($Log in @($Stdout,$Stderr)) { if (Test-Path -LiteralPath $Log) { $CombinedTail += Get-Content -LiteralPath $Log -Tail 80 -ErrorAction SilentlyContinue } }
$Done = 0; $BatchDone = 0; $BatchTotal = 0
$ProgressMatches = [regex]::Matches(($CombinedTail -join [Environment]::NewLine), 'epoch\(17\).*?(\d+)\s*/\s*(\d+)')
if ($ProgressMatches.Count -gt 0) { $LastMatch=$ProgressMatches[$ProgressMatches.Count-1]; $BatchDone=[int]$LastMatch.Groups[1].Value; $BatchTotal=[int]$LastMatch.Groups[2].Value; $Done=[Math]::Min($TotalFrames,$BatchDone*16) } elseif(Test-Path -LiteralPath $PredictionFile){$Done=$TotalFrames}
$Logs = @($Stdout,$Stderr) | Where-Object { Test-Path -LiteralPath $_ } | ForEach-Object { Get-Item -LiteralPath $_ }; $LastOutput = $Logs | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$GpuSummary = (& nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>$null) -join '; '
Write-Output "STATUS=$Status"; Write-Output "DONE_TOTAL=$Done/$TotalFrames BATCHES=$BatchDone/$BatchTotal"; Write-Output "PID=$PidValue"; if($Meta){Write-Output "START=$($Meta.start_time)"}; if($Process){Write-Output "COMMAND=$($Process.CommandLine)"}; if($LastOutput){Write-Output "LAST_OUTPUT=$($LastOutput.LastWriteTime.ToString('o')) FILE=$($LastOutput.FullName)"}; Write-Output "GPU=$GpuSummary"; Write-Output "PREDICTIONS=$PredictionFile EXISTS=$(Test-Path -LiteralPath $PredictionFile)"; Write-Output "STDOUT=$Stdout"; Write-Output "STDERR=$Stderr"; Write-Output '--- LOG TAIL ---'; $CombinedTail | Select-Object -Last 20





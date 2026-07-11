param([int]$TailLines = 25)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runnerRoot = Join-Path $repoRoot "artifacts\joint_training\yolomg_runner"
$pidPath = Join-Path $runnerRoot "train.pid"
$metaPath = Join-Path $runnerRoot "train.meta.json"
$meta = if (Test-Path -LiteralPath $metaPath) { Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json } else { $null }
$pidValue = if (Test-Path -LiteralPath $pidPath) { Get-Content -LiteralPath $pidPath | Select-Object -First 1 } else { $null }
$process = if ($pidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
$validProcess = $process -and $process.CommandLine -like "*train.py*" -and $process.CommandLine -like "*joint_nps_ard100.yaml*"
$resultsCsv = if ($meta) { Join-Path $meta.run_dir "results.csv" } else { $null }
$resultsTxt = if ($meta) { Join-Path $meta.run_dir "results.txt" } else { $null }
$done = 0
if ($resultsCsv -and (Test-Path -LiteralPath $resultsCsv)) { $done = [math]::Max(0, (Get-Content -LiteralPath $resultsCsv | Measure-Object -Line).Lines - 1) }
elseif ($resultsTxt -and (Test-Path -LiteralPath $resultsTxt)) { $done = (Get-Content -LiteralPath $resultsTxt | Where-Object { $_ -match '^\s*\d+/' } | Measure-Object).Count }
$latestCandidates = @($resultsCsv, $resultsTxt)
if ($meta) { $latestCandidates += @($meta.stdout, $meta.stderr, (Join-Path $meta.run_dir "weights\last.pt")) }
$latest = $latestCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | ForEach-Object { Get-Item -LiteralPath $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if ($validProcess) { Write-Output "RUNNING"; Write-Output "pid: $pidValue"; Write-Output "command: $($process.CommandLine)" } else { Write-Output "NOT RUNNING"; Write-Output "last_pid: $pidValue" }
Write-Output "done/total: $done/$(if($meta){$meta.epochs}else{'unknown'})"
Write-Output "start_time: $(if($meta){$meta.start_time}else{'unknown'})"
Write-Output "last_completed_unit: epoch=$done"
Write-Output "last_output_timestamp: $(if($latest){$latest.LastWriteTime.ToString('o')}else{'none'})"
Write-Output "stdout: $(if($meta){$meta.stdout}else{'none'})"
Write-Output "stderr: $(if($meta){$meta.stderr}else{'none'})"
Write-Output "== GPU signal =="
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits }
if ($meta -and (Test-Path -LiteralPath $meta.stdout)) { Write-Output "== stdout tail =="; Get-Content -LiteralPath $meta.stdout -Tail $TailLines }
if ($meta -and (Test-Path -LiteralPath $meta.stderr)) { Write-Output "== stderr tail =="; Get-Content -LiteralPath $meta.stderr -Tail $TailLines }

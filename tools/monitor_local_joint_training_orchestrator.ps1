param([int]$TailLines = 20)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$runRoot = Join-Path $repoRoot "artifacts\joint_training\orchestrator"
$pidPath = Join-Path $runRoot "orchestrator.pid"
$metaPath = Join-Path $runRoot "orchestrator.meta.json"
$phasePath = Join-Path $runRoot "phase.json"
$meta = if (Test-Path -LiteralPath $metaPath) { Get-Content -LiteralPath $metaPath -Raw | ConvertFrom-Json } else { $null }
$phase = if (Test-Path -LiteralPath $phasePath) { Get-Content -LiteralPath $phasePath -Raw | ConvertFrom-Json } else { $null }
$pidValue = if (Test-Path -LiteralPath $pidPath) { Get-Content -LiteralPath $pidPath | Select-Object -First 1 } else { $null }
$process = if ($pidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
$valid = $process -and $process.CommandLine -like "*local_joint_training_orchestrator_worker.ps1*"
$downloadState = Join-Path $(if($meta){$meta.dataset_root}else{"D:\URAP_local_datasets"}) ".download_state"
$plan = if (Test-Path -LiteralPath (Join-Path $downloadState "plan.json")) { @(Get-Content -LiteralPath (Join-Path $downloadState "plan.json") -Raw | ConvertFrom-Json) } else { @() }
$complete = @(Get-ChildItem -LiteralPath $downloadState -Filter "*.complete.json" -File -ErrorAction SilentlyContinue | Where-Object Name -ne "all.complete.json")

if ($valid) { Write-Output "RUNNING"; Write-Output "pid: $pidValue"; Write-Output "command: $($process.CommandLine)" } else { Write-Output "NOT RUNNING"; Write-Output "last_pid: $pidValue" }
Write-Output "done/total: $($complete.Count)/$($plan.Count) download units"
Write-Output "start_time: $(if($meta){$meta.start_time}else{'unknown'})"
Write-Output "last_completed_unit: phase=$(if($phase){$phase.phase}else{'unknown'})"
Write-Output "last_output_timestamp: $(if($phase){$phase.updated}else{'none'})"
Write-Output "stdout: $(if($meta){$meta.stdout}else{'none'})"
Write-Output "stderr: $(if($meta){$meta.stderr}else{'none'})"
Write-Output "== GPU signal =="
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits }
if ($meta -and (Test-Path -LiteralPath $meta.stdout)) { Write-Output "== stdout tail =="; Get-Content -LiteralPath $meta.stdout -Tail $TailLines }
if ($meta -and (Test-Path -LiteralPath $meta.stderr)) { Write-Output "== stderr tail =="; Get-Content -LiteralPath $meta.stderr -Tail $TailLines }

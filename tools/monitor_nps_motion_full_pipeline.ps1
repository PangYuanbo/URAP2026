param(
  [string]$RunnerRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness\pipeline_runner",
  [string]$RunId = "nps_motion_full_pipeline",
  [string]$ArtifactRoot = "C:\Users\aaron\Desktop\URAP\artifacts\nps_motion_robustness",
  [int]$TailLines = 30
)

$ErrorActionPreference = "Stop"
$pidFile = Join-Path $RunnerRoot "$RunId.pid"
$metaFile = Join-Path $RunnerRoot "$RunId.meta.txt"
$meta = @{}
if (Test-Path $metaFile) { foreach ($line in Get-Content $metaFile) { $index = $line.IndexOf('='); if ($index -gt 0) { $meta[$line.Substring(0, $index)] = $line.Substring($index + 1) } } }
$pidValue = if (Test-Path $pidFile) { Get-Content $pidFile | Select-Object -First 1 } else { $null }
$process = if ($pidValue) { Get-CimInstance Win32_Process -Filter "ProcessId = $pidValue" -ErrorAction SilentlyContinue } else { $null }
if ($process -and $process.CommandLine -like "*run_nps_motion_full_pipeline.ps1*") {
  Write-Host "RUNNING=true PID=$pidValue"
  Write-Host "PID_START=$((Get-Process -Id ([int]$pidValue)).StartTime)"
  Write-Host "PROCESS_COMMAND=$($process.CommandLine)"
} else { Write-Host "NOT RUNNING PID=$pidValue" }
Write-Host "== Active child jobs =="
$childPidFiles = Get-ChildItem $ArtifactRoot -Recurse -Filter '*.pid' -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch 'smoke' -and $_.FullName -ne $pidFile }
foreach ($childPidFile in $childPidFiles) {
  $childPid = Get-Content $childPidFile.FullName -ErrorAction SilentlyContinue | Select-Object -First 1
  $childProcess = if ($childPid) { Get-CimInstance Win32_Process -Filter "ProcessId = $childPid" -ErrorAction SilentlyContinue } else { $null }
  if ($childProcess) {
    $childStart = (Get-Process -Id ([int]$childPid) -ErrorAction SilentlyContinue).StartTime
    Write-Host "CHILD_RUNNER PID=$childPid START=$childStart PID_FILE=$($childPidFile.FullName)"
    Write-Host "CHILD_COMMAND=$($childProcess.CommandLine)"
    $grandchildren = Get-CimInstance Win32_Process -Filter "ParentProcessId = $childPid" -ErrorAction SilentlyContinue
    foreach ($grandchild in $grandchildren) {
      if ($grandchild.CommandLine -and $grandchild.CommandLine -notlike '*conhost.exe*') {
        Write-Host "WORKER_PID=$($grandchild.ProcessId) WORKER_COMMAND=$($grandchild.CommandLine)"
      }
    }
  }
}
$pipelineProgress = Join-Path $ArtifactRoot "pipeline_progress.json"
$payload = if (Test-Path $pipelineProgress) { Get-Content $pipelineProgress -Raw | ConvertFrom-Json } else { $null }
Write-Host "pipeline stage: $(if($payload){$payload.stage}else{'unknown'})"
Write-Host "pipeline status: $(if($payload){$payload.status}else{'unknown'})"
Write-Host "pipeline detail: $(if($payload){$payload.detail}else{'none'})"
$builderProgress = Join-Path "U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1" "progress.json"
$build = if (Test-Path $builderProgress) { Get-Content $builderProgress -Raw | ConvertFrom-Json } else { $null }
$integrityProgressPath = "U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1\original\integrity_progress.json"
$integrityProgress = if (Test-Path $integrityProgressPath) { Get-Content $integrityProgressPath -Raw | ConvertFrom-Json } else { $null }
$evalProgress = Join-Path $ArtifactRoot "model_evals\progress.json"
$eval = if (Test-Path $evalProgress) { Get-Content $evalProgress -Raw | ConvertFrom-Json } else { $null }
$trainResults = Join-Path $ArtifactRoot "yolomg_nps_train50\results.csv"
$epochs = if (Test-Path $trainResults) { [Math]::Max(0, (Get-Content $trainResults | Measure-Object -Line).Lines - 1) } else { 0 }
Write-Host "done/total: build=$(if($build){"$($build.done)/$($build.total)"}else{'0/?'}) train=$epochs/50 eval=$(if($eval){"$($eval.done)/$($eval.total)"}else{'0/15'})"
if ($integrityProgress) { Write-Host "integrity checked/total: $($integrityProgress.checked)/$($integrityProgress.total) split=$($integrityProgress.split)" }
$latest = @($pipelineProgress,$builderProgress,$evalProgress,$trainResults,$meta['stdout'],$meta['stderr']) | Where-Object { $_ -and (Test-Path $_) } | ForEach-Object { Get-Item $_ } | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "last output timestamp: $(if($latest){$latest.LastWriteTime}else{'none'})"
Write-Host "last completed unit: $(if($build){$build.last_completed_unit}else{'none'})"
Write-Host "stdout log: $($meta['stdout'])"
Write-Host "stderr log: $($meta['stderr'])"
Write-Host "== GPU signal =="
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits } else { Write-Host "nvidia-smi unavailable" }
if ($meta['stdout'] -and (Test-Path $meta['stdout'])) { Write-Host "== stdout tail =="; Get-Content $meta['stdout'] -Tail $TailLines }
if ($meta['stderr'] -and (Test-Path $meta['stderr'])) { Write-Host "== stderr tail =="; Get-Content $meta['stderr'] -Tail $TailLines }

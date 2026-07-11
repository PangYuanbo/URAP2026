param([string]$RunName = "ard100_short166_experiment_v1")

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$controlRoot = Join-Path $repoRoot "artifacts\samurai_runs\$RunName"
$logRoot = Join-Path $controlRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$statePath = Join-Path $controlRoot "state.json"
$modal = Join-Path $env:USERPROFILE ".local\bin\modal.exe"
$env:PYTHONUTF8 = "1"

function Write-State([string]$Stage, [hashtable]$Extra = @{}) {
    $payload = [ordered]@{ stage = $Stage; updated_at = (Get-Date).ToString("o") }
    foreach ($key in $Extra.Keys) { $payload[$key] = $Extra[$key] }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding utf8
}

function Get-RemoteManifest([string]$Split) {
    $remotePath = "ARD100_SAMURAI_SHORT166/$($Split)_v1/manifest.json"
    $downloadRoot = Join-Path $controlRoot "manifest_probe"
    New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
    $localPath = Join-Path $downloadRoot "$Split.json"
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $modal volume get --force urap-ard100-samurai-short166-v1 $remotePath $localPath 2>$null | Out-Null
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath $localPath)) { return $null }
    try { return Get-Content -LiteralPath $localPath -Raw | ConvertFrom-Json } catch { return $null }
}

function Test-JobSucceeded($Job) {
    $stderr = Get-Item -LiteralPath $Job.stderr_log -ErrorAction SilentlyContinue
    $stdoutText = if (Test-Path -LiteralPath $Job.stdout_log) { Get-Content -LiteralPath $Job.stdout_log -Raw } else { "" }
    return (-not ($stderr -and $stderr.Length -gt 0) -and $stdoutText -match "App completed")
}

function Start-ModalJob([string]$Name, [string[]]$Arguments) {
    $stdout = Join-Path $logRoot "$Name.stdout.log"
    $stderr = Join-Path $logRoot "$Name.stderr.log"
    $process = Start-Process -FilePath $modal -ArgumentList $Arguments -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $process.Id | Set-Content -LiteralPath (Join-Path $controlRoot "$Name.pid") -Encoding ascii
    $meta = [ordered]@{ pid = $process.Id; started_at = (Get-Date).ToString("o"); command = "$modal $($Arguments -join ' ')"; stdout_log = $stdout; stderr_log = $stderr }
    $meta | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $controlRoot "$Name.meta.json") -Encoding utf8
    return $meta
}

$required = [ordered]@{ train = 55; val = 10; test = 35 }
while ($true) {
    $ready = @()
    $manifests = [ordered]@{}
    foreach ($split in $required.Keys) {
        $manifest = Get-RemoteManifest $split
        if ($manifest) { $manifests[$split] = $manifest }
        if ($manifest -and [int]$manifest.source_video_count -eq [int]$required[$split]) { $ready += $split }
    }
    Write-State "waiting_for_dataset" @{ done = $ready.Count; total = $required.Count; ready = $ready; manifests = $manifests }
    if ($ready.Count -eq $required.Count) { break }
    Start-Sleep -Seconds 60
}

$jobs = [ordered]@{}
$jobs.train = Start-ModalJob "train" @("run", "tools\modal_train_ard100_short166.py")
foreach ($run in @("image_box_zero_shot", "sam2_video_zero_shot", "samurai_zero_shot")) {
    $jobs[$run] = Start-ModalJob $run @("run", "tools\modal_eval_ard100_short166.py", "--run", $run, "--shards", "4")
    while ($true) {
        $evalPid = [int]$jobs[$run].pid
        $evalProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$evalPid" -ErrorAction SilentlyContinue
        Write-State "zero_shot_running" @{ current = $run; jobs = $jobs }
        if (-not ($evalProcess -and $evalProcess.CommandLine -like "*modal_eval_ard100_short166.py*")) { break }
        Start-Sleep -Seconds 60
    }
}
Write-State "jobs_started" @{ jobs = $jobs }

while ($true) {
    $running = @()
    foreach ($name in $jobs.Keys) {
        $pid = [int]$jobs[$name].pid
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$pid" -ErrorAction SilentlyContinue
        if ($process -and $process.CommandLine -like "*modal*") { $running += $name }
    }
    Write-State "jobs_running" @{ running = $running; done = $jobs.Count - $running.Count; total = $jobs.Count; jobs = $jobs }
    if ($running.Count -eq 0) { break }
    Start-Sleep -Seconds 60
}

if (-not (Test-JobSucceeded $jobs.train)) {
    Write-State "failed" @{ failures = @("train"); jobs = $jobs }
    throw "Training failed or did not produce a completed Modal run"
}
foreach ($run in @("sam2_video_finetuned", "samurai_finetuned")) {
    $jobs[$run] = Start-ModalJob $run @("run", "tools\modal_eval_ard100_short166.py", "--run", $run, "--shards", "4")
    while ($true) {
        $evalPid = [int]$jobs[$run].pid
        $evalProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$evalPid" -ErrorAction SilentlyContinue
        Write-State "finetuned_eval_running" @{ current = $run; jobs = $jobs }
        if (-not ($evalProcess -and $evalProcess.CommandLine -like "*modal_eval_ard100_short166.py*")) { break }
        Start-Sleep -Seconds 60
    }
}

$failures = @()
foreach ($name in $jobs.Keys) {
    if (-not (Test-JobSucceeded $jobs[$name])) { $failures += $name }
}
if ($failures.Count -gt 0) {
    Write-State "failed" @{ failures = $failures; jobs = $jobs }
    throw "Experiment jobs failed or incomplete: $($failures -join ', ')"
}
Write-State "completed" @{ done = $jobs.Count; total = $jobs.Count; jobs = $jobs }

param(
    [string]$DatasetRoot = "D:\URAP_local_datasets",
    [string]$NpsRoot = "",
    [string]$ArdRoot = "",
    [string]$ManifestRoot = "",
    [switch]$SkipDownloadWait,
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stateRoot = Join-Path $repoRoot "artifacts\joint_training\orchestrator"
$phasePath = Join-Path $stateRoot "phase.json"
$downloadRunRoot = Join-Path $repoRoot "artifacts\local_joint_dataset_download"
$downloadPidPath = Join-Path $downloadRunRoot "download.pid"
$downloadStateRoot = Join-Path $DatasetRoot ".download_state"
$requiredDownloadUnits = @(
    "nps_images_train", "nps_images2_train", "nps_labels_train",
    "ard_images_train", "ard_images2_train", "ard_labels_train",
    "nps_images_val", "nps_images2_val", "nps_labels_val",
    "ard_images_val", "ard_images2_val", "ard_labels_val"
)
New-Item -ItemType Directory -Force -Path $stateRoot | Out-Null
if (-not $NpsRoot) { $NpsRoot = Join-Path $DatasetRoot "NPS_YOLOMG" }
if (-not $ArdRoot) { $ArdRoot = Join-Path $DatasetRoot "ARD100_YOLOMG" }
if (-not $ManifestRoot) { $ManifestRoot = Join-Path $DatasetRoot "joint_yolomg" }

function Set-Phase([string]$Phase, [hashtable]$Extra = @{}) {
    $payload = @{ phase = $Phase; updated = (Get-Date).ToString("o") }
    foreach ($key in $Extra.Keys) { $payload[$key] = $Extra[$key] }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $phasePath -Encoding UTF8
}

if (-not $SkipDownloadWait) {
    Set-Phase "waiting_for_download"
    while (@($requiredDownloadUnits | Where-Object { -not (Test-Path -LiteralPath (Join-Path $downloadStateRoot "$_.complete.json")) }).Count -gt 0) {
        $downloadPid = if (Test-Path -LiteralPath $downloadPidPath) { Get-Content -LiteralPath $downloadPidPath | Select-Object -First 1 } else { $null }
        $downloadProcess = if ($downloadPid) { Get-CimInstance Win32_Process -Filter "ProcessId = $downloadPid" -ErrorAction SilentlyContinue } else { $null }
        if (-not ($downloadProcess -and $downloadProcess.CommandLine -like "*download_local_joint_datasets_worker.ps1*")) {
            Set-Phase "blocked_download_not_running" @{ download_pid = $downloadPid }
            throw "Dataset download is NOT RUNNING and is incomplete"
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

Set-Phase "building_manifests"
$python = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\.venv\Scripts\python.exe"
$builder = Join-Path $PSScriptRoot "build_joint_yolomg_dataset.py"
& $python $builder --nps-root $NpsRoot --ard-root $ArdRoot --output $ManifestRoot --skip-incomplete-pairs
if ($LASTEXITCODE -ne 0) { Set-Phase "blocked_manifest_failure"; throw "Joint manifest build failed" }

$yolomgRoot = Join-Path $repoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG"
$smokeRun = Join-Path $repoRoot "artifacts\joint_training\yolomg_smoke"
$smokeLogRoot = Join-Path $stateRoot "logs"
New-Item -ItemType Directory -Force -Path $smokeLogRoot | Out-Null
$smokeStdout = Join-Path $smokeLogRoot "smoke.out.txt"
$smokeStderr = Join-Path $smokeLogRoot "smoke.err.txt"
$smokeArgs = @(
    "train.py", "--data", (Join-Path $ManifestRoot "smoke\joint_nps_ard100_smoke.yaml"),
    "--cfg", (Join-Path $yolomgRoot "models\NPS_uav_s.yaml"), "--weights", (Join-Path $yolomgRoot "yolov5s.pt"),
    "--epochs", "1", "--batch-size", "1", "--imgsz", "1280", "--device", "0", "--workers", "0",
    "--project", (Split-Path $smokeRun -Parent), "--name", (Split-Path $smokeRun -Leaf), "--exist-ok"
)
Set-Phase "smoke_running" @{ stdout = $smokeStdout; stderr = $smokeStderr }
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$smoke = Start-Process -FilePath $python -ArgumentList $smokeArgs -WorkingDirectory $yolomgRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput $smokeStdout -RedirectStandardError $smokeStderr
$smoke.Id | Set-Content -LiteralPath (Join-Path $stateRoot "smoke.pid") -Encoding ASCII
$smoke.WaitForExit()
$smoke.Refresh()
$smokeCheckpoint = Join-Path $smokeRun "weights\last.pt"
$smokeExitCode = $smoke.ExitCode
if (-not (Test-Path -LiteralPath $smokeCheckpoint) -or ($null -ne $smokeExitCode -and $smokeExitCode -ne 0)) {
    Set-Phase "blocked_smoke_failure" @{ smoke_pid = $smoke.Id; exit_code = $smokeExitCode; stdout = $smokeStdout; stderr = $smokeStderr }
    throw "YOLOMG smoke failed, exit=$smokeExitCode checkpoint=$smokeCheckpoint"
}

Set-Phase "starting_full_training" @{ smoke_pid = $smoke.Id; smoke_exit_code = $smoke.ExitCode }
& (Join-Path $PSScriptRoot "start_yolomg_joint_train_detached.ps1") -DataYaml (Join-Path $ManifestRoot "joint_nps_ard100.yaml") -BatchSize 2 -Workers 2 -Epochs 20
if ($LASTEXITCODE -ne 0) { Set-Phase "blocked_full_start_failure"; throw "Full training start failed" }
& (Join-Path $PSScriptRoot "start_yolomg_checkpoint_snapshot_detached.ps1") -RunDir (Join-Path $repoRoot "artifacts\joint_training\yolomg_nps_ard100_e20") -IntervalSeconds 7200
if ($LASTEXITCODE -ne 0) { Set-Phase "blocked_snapshot_start_failure"; throw "Checkpoint snapshot watcher start failed" }
Set-Phase "full_training_started" @{ epochs = 20; snapshot_interval_seconds = 7200 }

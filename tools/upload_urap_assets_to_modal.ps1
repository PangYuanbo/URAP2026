param(
    [string]$RepoRoot = "C:\Users\aaron\Desktop\URAP",
    [string]$StateDir = "artifacts\modal_urap_upload",
    [string]$OnlyVolume = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path -LiteralPath $RepoRoot).Path
$StateDir = Join-Path $RepoRoot $StateDir
$CompletedDir = Join-Path $StateDir "completed"
$ProgressPath = Join-Path $StateDir "progress.json"
New-Item -ItemType Directory -Force -Path $CompletedDir | Out-Null

$env:PATH = "$HOME\.local\bin;$env:PATH"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$jobs = @(
    @{ Id="nps_allframes"; Volume="urap-nps-formatted-v1"; Local="U:\URAP_datasets\TransVisDrone\NPS\AllFrames"; Remote="/NPS/" },
    @{ Id="nps_labels"; Volume="urap-nps-formatted-v1"; Local="U:\URAP_datasets\TransVisDrone\NPS\NPSvisdroneStyle"; Remote="/NPS/" },
    @{ Id="nps_samurai"; Volume="urap-nps-formatted-v1"; Local="U:\URAP_datasets\TransVisDrone\NPS\SAMURAI"; Remote="/NPS/" },
    @{ Id="nps_videos_metadata"; Volume="urap-nps-formatted-v1"; Local="U:\URAP_datasets\TransVisDrone\NPS\Videos"; Remote="/NPS/" },

    @{ Id="motion_original"; Volume="urap-nps-motion-original-v1"; Local="U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1\original"; Remote="/motion_v1/" },
    @{ Id="motion_slow_0p5"; Volume="urap-nps-motion-variants-v1"; Local="U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1\slow_0p5"; Remote="/motion_v1/" },
    @{ Id="motion_fast_2x"; Volume="urap-nps-motion-variants-v1"; Local="U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1\fast_2x"; Remote="/motion_v1/" },
    @{ Id="motion_accelerate_g2"; Volume="urap-nps-motion-variants-v1"; Local="U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1\accelerate_g2"; Remote="/motion_v1/" },
    @{ Id="motion_decelerate_g2"; Volume="urap-nps-motion-variants-v1"; Local="U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1\decelerate_g2"; Remote="/motion_v1/" },

    @{ Id="ard100_raw"; Volume="urap-ard100-raw-v1"; Local="U:\URAP_datasets\ARD100"; Remote="/datasets/" },
    @{ Id="ard100_train_images"; Volume="urap-ard100-yolomg-train-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG\images\train"; Remote="/ARD100_YOLOMG/images/" },
    @{ Id="ard100_train_images2"; Volume="urap-ard100-yolomg-train-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG\images2\train"; Remote="/ARD100_YOLOMG/images2/" },
    @{ Id="ard100_train_labels"; Volume="urap-ard100-yolomg-train-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG\labels\train"; Remote="/ARD100_YOLOMG/labels/" },
    @{ Id="ard100_val_images"; Volume="urap-ard100-yolomg-eval-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG\images\val"; Remote="/ARD100_YOLOMG/images/" },
    @{ Id="ard100_val_images2"; Volume="urap-ard100-yolomg-eval-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG\images2\val"; Remote="/ARD100_YOLOMG/images2/" },
    @{ Id="ard100_val_labels"; Volume="urap-ard100-yolomg-eval-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG\labels\val"; Remote="/ARD100_YOLOMG/labels/" },
    @{ Id="ard100_test_images"; Volume="urap-ard100-yolomg-eval-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG\images\test"; Remote="/ARD100_YOLOMG/images/" },
    @{ Id="ard100_test_images2"; Volume="urap-ard100-yolomg-eval-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG\images2\test"; Remote="/ARD100_YOLOMG/images2/" },
    @{ Id="ard100_test_labels"; Volume="urap-ard100-yolomg-eval-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG\labels\test"; Remote="/ARD100_YOLOMG/labels/" },
    @{ Id="ard100_annotations"; Volume="urap-ard100-yolomg-annotations-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG\annotations"; Remote="/ARD100_YOLOMG/" },
    @{ Id="ard100_test_layout1"; Volume="urap-yolomg-eval-extras-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG_TEST"; Remote="/datasets/" },
    @{ Id="ard100_test_layout2"; Volume="urap-yolomg-eval-extras-v1"; Local="U:\URAP_datasets\ARD100_YOLOMG_TEST2"; Remote="/datasets/" },
    @{ Id="yolomg_eval"; Volume="urap-yolomg-eval-extras-v1"; Local="U:\URAP_datasets\YOLOMG_eval"; Remote="/datasets/" },

    @{ Id="weight_tvd_aot_best"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\AOT\image_size_1280_YOLOXL_3_frames_AOT_with_yolo_weights_end_to_end\weights\best.pt"); Remote="/TransVisDrone/AOT/best.pt" },
    @{ Id="weight_tvd_aot_last"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\AOT\image_size_1280_YOLOXL_3_frames_AOT_with_yolo_weights_end_to_end\weights\last.pt"); Remote="/TransVisDrone/AOT/last.pt" },
    @{ Id="weight_tvd_nps_best"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\best.pt"); Remote="/TransVisDrone/NPS/best.pt" },
    @{ Id="weight_tvd_nps_last"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\NPS\image_size_1280_temporal_YOLO5l_5_frames_NPS_end_to_end_skip_0\weights\last.pt"); Remote="/TransVisDrone/NPS/last.pt" },
    @{ Id="weight_tvd_fl_best"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\FL\image_size_1280_temporal_YOLO5L_5_frames_FL_end_to_end\weights\best.pt"); Remote="/TransVisDrone/FL/best.pt" },
    @{ Id="weight_tvd_fl_last"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "papers\TransVisDrone\pretrained\TransVisDrone_weights\runs\train\FL\image_size_1280_temporal_YOLO5L_5_frames_FL_end_to_end\weights\last.pt"); Remote="/TransVisDrone/FL/last.pt" },
    @{ Id="weight_yolomg_ard100_1280_best"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\ARD100_mask32-1280_uavs\weights\best.pt"); Remote="/YOLOMG/ARD100_mask32-1280/best.pt" },
    @{ Id="weight_yolomg_ard100_640_best"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\ARD100_mask32-640_uavs\weights\best.pt"); Remote="/YOLOMG/ARD100_mask32-640/best.pt" },
    @{ Id="weight_yolomg_ard100_e50a_best"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260217_084905\weights\best.pt"); Remote="/YOLOMG/yolomg_ard100_e50_b4_img1280_20260217/best.pt" },
    @{ Id="weight_yolomg_ard100_e50a_last"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260217_084905\weights\last.pt"); Remote="/YOLOMG/yolomg_ard100_e50_b4_img1280_20260217/last.pt" },
    @{ Id="weight_yolomg_ard100_e50b_best"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt"); Remote="/YOLOMG/yolomg_ard100_e50_b4_img1280_20260221/best.pt" },
    @{ Id="weight_yolomg_ard100_e50b_last"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\last.pt"); Remote="/YOLOMG/yolomg_ard100_e50_b4_img1280_20260221/last.pt" },
    @{ Id="weight_yolov5s"; Volume="urap-model-weights-v1"; Local=(Join-Path $RepoRoot "URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\yolov5s.pt"); Remote="/YOLOMG/pretrained/yolov5s.pt" },
    @{ Id="weight_samurai_base_plus"; Volume="urap-model-weights-v1"; Local="U:\URAP_models\samurai\sam2.1_hiera_base_plus.pt"; Remote="/SAMURAI/sam2.1_hiera_base_plus.pt" },
    @{ Id="weight_samurai_tiny"; Volume="urap-model-weights-v1"; Local="U:\URAP_models\samurai\sam2.1_hiera_tiny.pt"; Remote="/SAMURAI/sam2.1_hiera_tiny.pt" },
    @{ Id="weight_samurai_bbox"; Volume="urap-model-weights-v1"; Local="U:\URAP_models\samurai\bbox_readout_finetuned1.pt"; Remote="/SAMURAI/bbox_readout_finetuned1.pt" },

    @{ Id="experiment_artifacts"; Volume="urap-code-artifacts-v1"; Local=(Join-Path $RepoRoot "artifacts\nps_motion_robustness"); Remote="/artifacts/" },
    @{ Id="repo_tools"; Volume="urap-code-artifacts-v1"; Local=(Join-Path $RepoRoot "tools"); Remote="/repo/" },
    @{ Id="repo_qstr"; Volume="urap-code-artifacts-v1"; Local=(Join-Path $RepoRoot "qstr_dronedet"); Remote="/repo/" },
    @{ Id="repo_tests"; Volume="urap-code-artifacts-v1"; Local=(Join-Path $RepoRoot "tests"); Remote="/repo/" },
    @{ Id="repo_docs"; Volume="urap-code-artifacts-v1"; Local=(Join-Path $RepoRoot "docs"); Remote="/repo/" }
)

$jobs = @($jobs | Sort-Object `
    @{ Expression = { if ($_.Id -like "weight_*") { 0 } elseif ($_.Id -like "repo_*" -or $_.Id -eq "experiment_artifacts") { 1 } else { 2 } } }, `
    @{ Expression = { $_.Id } })

if ($OnlyVolume) {
    $jobs = @($jobs | Where-Object { $_.Volume -eq $OnlyVolume })
    if (-not $jobs.Count) { throw "No upload jobs found for volume: $OnlyVolume" }
    $safeVolume = $OnlyVolume -replace "[^A-Za-z0-9_.-]", "_"
    $ProgressPath = Join-Path $StateDir ("progress_" + $safeVolume + ".json")
}

function Write-ProgressState([int]$Index, [string]$Status, [hashtable]$Job, [string]$Message) {
    $completed = @($jobs | Where-Object { Test-Path -LiteralPath (Join-Path $CompletedDir ($_.Id + ".json")) }).Count
    $state = [ordered]@{
        updated = (Get-Date).ToString("o")
        status = $Status
        done = $completed
        total = $jobs.Count
        worker_volume = $OnlyVolume
        current_index = $Index
        current_job = if ($Job) { $Job.Id } else { $null }
        volume = if ($Job) { $Job.Volume } else { $null }
        local = if ($Job) { $Job.Local } else { $null }
        remote = if ($Job) { $Job.Remote } else { $null }
        message = $Message
    }
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ProgressPath -Encoding utf8
}

Write-ProgressState -Index 0 -Status "running" -Job $null -Message "starting"
for ($index = 0; $index -lt $jobs.Count; $index++) {
    $job = $jobs[$index]
    $marker = Join-Path $CompletedDir ($job.Id + ".json")
    if ((Test-Path -LiteralPath $marker) -and -not $Force) {
        Write-ProgressState -Index ($index + 1) -Status "running" -Job $job -Message "already completed; skipped"
        continue
    }
    if (-not (Test-Path -LiteralPath $job.Local)) {
        throw "Missing local source for $($job.Id): $($job.Local)"
    }
    Write-ProgressState -Index ($index + 1) -Status "running" -Job $job -Message "uploading"
    $arguments = @("volume", "put", "-f", $job.Volume, $job.Local, $job.Remote)
    & modal @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "modal upload failed for $($job.Id), exit=$LASTEXITCODE"
    }
    [ordered]@{
        completed = (Get-Date).ToString("o")
        id = $job.Id
        volume = $job.Volume
        local = $job.Local
        remote = $job.Remote
    } | ConvertTo-Json | Set-Content -LiteralPath $marker -Encoding utf8
    Write-ProgressState -Index ($index + 1) -Status "running" -Job $job -Message "completed"
}
Write-ProgressState -Index $jobs.Count -Status "complete" -Job $null -Message "all uploads completed"

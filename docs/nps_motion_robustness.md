# NPS Motion Intervention Robustness

This pipeline builds label-preserving temporal interventions without modifying the source NPS dataset at `U:\URAP_datasets\TransVisDrone\NPS`.

## Full Pipeline

Start the complete recoverable sequence with one detached command:

```powershell
.\tools\start_nps_motion_full_pipeline_detached.ps1
.\tools\monitor_nps_motion_full_pipeline.ps1
```

The orchestrator builds the original train/val/test package, starts YOLOMG NPS-50 training and the four changed test datasets, waits for both branches, evaluates three models over five interventions, and generates the final report. Completed clip, model, and intervention markers are reused after an explicitly observed restart; failed jobs are not silently restarted.

## Dataset Build

The full detached build creates the original control for train/val/test and all five test interventions:

```powershell
.\tools\start_nps_motion_interventions_detached.ps1
.\tools\monitor_nps_motion_interventions.ps1
```

Outputs are written under `U:\URAP_datasets\TransVisDrone\NPS_interventions\motion_v1` with separate TransVisDrone and YOLOMG layouts. Every clip has a JSONL frame manifest, completion marker, audit image, fallback rate, and integrity report.

`original` uses hard links and must pass `original_source_equivalent=true`. `slow_0p5` uses bidirectional OpenCV DIS interpolation. `fast_2x` samples every second source frame. `accelerate_g2` and `decelerate_g2` use the frozen quadratic time maps from the experiment protocol.

## YOLOMG NPS Training

No locally verifiable NPS-trained YOLOMG checkpoint is present. After the original train/val dataset build completes, start the budget reproduction:

```powershell
.\tools\start_nps_yolomg_train50_detached.ps1
.\tools\monitor_nps_yolomg_train50.ps1
```

This run uses the repository's NPS model config, `yolov5s.pt`, image size 1280, batch size 8, and 50 epochs. It is intentionally labeled as a budget reproduction rather than the README's 100-epoch recipe. A stopped run is resumed only with the explicit `-ResumeExisting` switch.

## Model Evaluation

After the dataset and native checkpoint are available:

```powershell
.\tools\start_nps_motion_model_evals_detached.ps1
.\tools\monitor_nps_motion_model_evals.ps1
```

The runner evaluates TransVisDrone, YOLOMG NPS-50, and the existing YOLOMG ARD100 checkpoint. The ARD100 model is marked as a cross-domain control. All runs save prediction labels with confidence so the final report uses one metric implementation.

## Unified Report

```powershell
& .\papers\TransVisDrone\.venv\Scripts\python.exe .\tools\summarize_nps_motion_robustness.py
```

The report writes dataset-, clip-, and frame-level CSV files, paired clip bootstrap intervals, anchor/synthetic slowdown slices, mAP retention curves, and worst-clip timelines. Fractional timestamps use the nearest original source frame for the matched-control distribution. High-motion limitation is supported only for a primary model with at least five common clips, when the matched mAP@0.5 drop exceeds 10%, the paired 95% interval is above zero, and the original source-equivalence gate passes. The ARD100 checkpoint never triggers the primary claim.

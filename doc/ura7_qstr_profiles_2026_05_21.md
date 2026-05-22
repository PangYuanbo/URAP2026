# URA-7 QSTR Operating Profiles

Date: 2026-05-21

## Goal

Freeze the train/adapt-selected QSTR profiles from URA-6 before any final frozen10 validation. Frozen10 remains a final validation set only.

## Stable Profile

Stable is the default low-FP profile for normal sequences where the primary YOLO-P2 detector already provides usable candidates.

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_stable_profile.ps1 `
  -Video D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\raw_videos\test\visible\20190925_111757_1_5\visible.mp4 `
  -Out runs\profiles\stable_example `
  -Device 0 `
  -MaxFrames 60
```

Frozen profile settings:

- tracklet classifier: `runs\profiles\tracklet_train_eval_20260521_154002\tracklet_mlp_v2_hardtiny_aug.pt`
- classifier threshold: `0.95`
- tracklet promotion: disabled
- verified objectness boost: disabled

URA-6 train/adapt result:

| output | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|
| raw | 109 | 864 | 11 | 0.1120 | 0.9083 |
| stable selected | 109 | 615 | 11 | 0.1506 | 0.9083 |

## Hard-Recovery Profile

Hard-recovery is an experimental recovery profile for hard tiny, low-objectness, fallback, static/hovering, fast-motion, or bad-alignment cases. It is not the default profile.

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_hard_recovery_profile.ps1 `
  -Video D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\raw_videos\test\visible\20190925_111757_1_7\visible.mp4 `
  -Out runs\profiles\hard_recovery_example `
  -Device 0 `
  -MaxFrames 60
```

Frozen profile settings:

- tracklet classifier: `runs\profiles\tracklet_train_eval_20260521_154002\tracklet_mlp_v2_hardtiny_aug.pt`
- classifier threshold: `0.50`
- tracklet promotion: enabled
- promotion score floor: `0.30`
- promotion max background: `0.55`

URA-6 train/adapt result:

| output | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|
| raw | 109 | 864 | 11 | 0.1120 | 0.9083 |
| hard-recovery selected | 112 | 945 | 8 | 0.1060 | 0.9333 |

The hard-recovery candidate improved recall by 2.5 points but did not meet the 5-point target, so final frozen10 validation must decide whether it remains diagnostic-only.

## Frozen10 Validation Command

Use this after URA-7 only. Do not change thresholds based on frozen10.

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_frozen10_profile_benchmark.ps1 `
  -HeldoutRoot D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq `
  -AnnotationsCsv D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\annotations\qstr_real_boxes.csv `
  -OutRoot runs\profiles\frozen10_ura8_final_20260521 `
  -Device 0 `
  -MaxVideos 10 `
  -MaxFrames 60
```

The benchmark reports `hard_recovery_raw`, `stable`, and `hard_recovery` in `profile_benchmark_summary.json`.


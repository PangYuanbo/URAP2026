# Public Dataset Scene-Recovery Plan

## Goal

Use public datasets to pretrain and stress the QSTR Stage B / tracklet scene-recovery path without contaminating DJI `121806` held-out.

The target failure mode is:

```text
Stage A proposes a true tiny drone candidate,
strict Stage B suppresses it as background,
the target is persistent across frames,
and GT confirms it is a drone.
```

## Dataset Roles

| dataset | role | use as drone positive? | notes |
| --- | --- | --- | --- |
| Anti-UAV300 | persistent UAV tracklets | yes | best current public source for continuous UAV boxes |
| ARD100 | tiny RGB drone positives | yes | useful for small RGB drone candidate positives |
| AOT | airborne object candidate/unknown | no | useful for Stage A and unknown airborne negatives, not drone identity |
| Drone-vs-Bird / USC-GRAD-STDdb | bird/tiny hard negatives | only when label is drone | requires separate access/application |

## Current Local Availability

Available under `D:\datasets`:

- `D:\datasets\Anti-UAV300`
- `D:\datasets\ARD100`
- `D:\datasets\AOT_sample`
- `D:\datasets\AOT_train_adapt_pool_v2`

Anti-UAV QSTR subsets already exist:

| split | annotations | boxes |
| --- | --- | ---: |
| train20 | `D:\datasets\Anti-UAV300\qstr_train_visible_20seq\annotations\qstr_real_boxes.csv` | 1600 |
| val5 | `D:\datasets\Anti-UAV300\qstr_train_visible_val_5seq\annotations\qstr_real_boxes.csv` | 400 |
| adapt5 | `D:\datasets\Anti-UAV300\qstr_adapt_test_visible_5seq\annotations\qstr_real_boxes.csv` | 400 |
| heldout10 | `D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\annotations\qstr_real_boxes.csv` | 737 |

## Anti-UAV Scene-Recovery Runner

Added:

- `tools/start_antiuav_scene_recovery_profiles_detached.ps1`
- `tools/monitor_antiuav_scene_recovery_profiles.ps1`

Default behavior:

- Stage A detector: `D:\datasets\stage_a_mixed\runs\ard100_dji_aot_yolo_p2_v2\yolo_p2_candidate\weights\best.pt`
- recall-oriented Stage B: first DJI hard-negative repair model;
- strict Stage B: dense DJI hard-negative repair model;
- tile size/stride: `256/128`;
- frame stride: `5`, matching the Anti-UAV exported annotation stride;
- max frames per video: `0`, meaning run through the annotated frame range.

Run train split:

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_antiuav_scene_recovery_profiles_detached.ps1 -Split train20
```

Monitor:

```powershell
powershell -ExecutionPolicy Bypass -File tools\monitor_antiuav_scene_recovery_profiles.ps1 -Split train20
```

After train output completes, run validation split:

```powershell
powershell -ExecutionPolicy Bypass -File tools\start_antiuav_scene_recovery_profiles_detached.ps1 -Split val5
```

## Mining Discipline

Use Anti-UAV train/val for public-data experiments only. Do not select thresholds on DJI `121806`.

Recommended sequence:

1. Generate Anti-UAV train20 recall/strict paired outputs.
2. Generate Anti-UAV val5 recall/strict paired outputs.
3. Train scene gate on Anti-UAV train20 with `--sample-mode gt_suppressed_candidate`.
4. Select only on Anti-UAV val5 or a mixed non-held-out calibration split.
5. If it improves public and DJI calibration metrics, run one fixed DJI held-out check.

## Decision Boundary

Anti-UAV can improve persistent tiny-drone pretraining, but it cannot replace new DJI non-held-out data. If a public-data gate does not improve DJI calibration, do not evaluate it on `121806`.

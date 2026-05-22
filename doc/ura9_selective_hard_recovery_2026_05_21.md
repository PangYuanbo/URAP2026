# URA-9 Selective Hard-Recovery Gate

Date: 2026-05-21

## Goal

Try to combine stable-profile FP control with hard-recovery recall using a stricter tracklet-level promotion gate and a per-sequence promotion budget.

## Implementation

Added optional selective tracklet promotion to the post-infer tracklet filter.

Selective promotion requires:

- candidate tracklet classified as drone;
- branch drone evidence above the promotion threshold;
- background evidence below the promotion threshold;
- temporal evidence above crop by a configured margin;
- temporal evidence not overwhelmed by background;
- minimum tracklet length;
- recovery-source support unless explicitly disabled;
- per-sequence maximum promoted tracklets.

The default hard-recovery profile remains the URA-7 profile. Selective promotion is explicit:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_hard_recovery_profile.ps1 `
  -Video D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\raw_videos\test\visible\20190925_111757_1_7\visible.mp4 `
  -Out runs\profiles\hard_recovery_selective_example `
  -Device 0 `
  -MaxFrames 60 `
  -EnableSelectiveTrackletPromotion
```

## Train/Adapt Calibration

The best train/adapt selective candidate was:

```text
tracklet threshold: 0.50
promotion score floor: 0.24
promotion max background: 0.60
selective min temporal-crop delta: 0.05
selective min temporal-background margin: -0.10
selective max tracklet objectness: 1.0
selective min temporal gain rate: 0.20
selective max promoted tracklets per sequence: 1
```

Train/adapt result:

| profile | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|
| raw hard-recovery | 109 | 864 | 11 | 0.1120 | 0.9083 |
| selective candidate | 112 | 863 | 8 | 0.1149 | 0.9333 |

This looked acceptable on train/adapt: TP +3, FP -1.

## Frozen10 Validation

Final frozen10 validation did not pass:

| profile | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|
| hard_recovery_raw | 37 | 682 | 17 | 0.0515 | 0.6852 |
| hard_recovery_selective | 35 | 632 | 19 | 0.0525 | 0.6481 |
| stable | 28 | 257 | 26 | 0.0982 | 0.5185 |

Per-frame result:

| output | frame success | false_positive_no_gt frames |
|---|---:|---:|
| raw hard-recovery | 0.6852 | 393 |
| selective hard-recovery | 0.6481 | 376 |

## Decision

URA-9 is not solved.

The selective gate reduces FP, but it suppresses too many true positives on frozen10. It should remain an experimental switch, not the default hard-recovery profile.

URA-5 should stay open.

## Artifacts

```text
reports\URAP-UAV\ura9_selective_tracklet_sweep_train_adapt_objectness1_20260521\tracklet_filter_sweep.csv
runs\profiles\frozen10_ura9_selective_objectness1_final_20260521\profile_benchmark_summary.csv
reports\URAP-UAV\ura9_frozen10_selective_objectness1_hard_recovery_20260521\raw_frame_timeline.png
reports\URAP-UAV\ura9_frozen10_selective_objectness1_hard_recovery_20260521\filtered_frame_timeline.png
```

## Next Direction

The current tracklet aggregate features are not enough to separate recoverable tiny drones from FP tracks. Next work should add localization-aware and temporal-shape features before another calibration pass:

- best stable-vs-hard IoU overlap per tracklet;
- per-tracklet score trend, not only mean values;
- consecutive-hit length and gaps;
- distance from primary YOLO candidates;
- per-frame background/artifact dominance streaks.


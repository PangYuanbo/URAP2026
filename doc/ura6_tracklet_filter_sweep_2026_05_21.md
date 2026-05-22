# URA-6 Tracklet Filter Threshold Sweep

Date: 2026-05-21

## Goal

Run a train/adapt-only threshold sweep for tracklet promotion and rejection. This is calibration work for the tracklet filter and must not tune on frozen10.

## Inputs

- Run roots:
  - `runs/profiles/tracklet_train_eval_20260521_154002/train5_60f`
  - `runs/profiles/tracklet_train_eval_20260521_154002/adapt5_60f`
- GT CSV: `runs/profiles/tracklet_train_eval_20260521_154002/train_adapt_gt.csv`
- Tracklet classifier: `runs/profiles/tracklet_train_eval_20260521_154002/tracklet_mlp_v2_hardtiny_aug.pt`
- Output: `reports/URAP-UAV/ura6_tracklet_filter_sweep_train_adapt_20260521_v2`

## Command

```powershell
python -m qstr_dronedet.cli sweep-tracklet-filter `
  --run-roots runs\profiles\tracklet_train_eval_20260521_154002\train5_60f runs\profiles\tracklet_train_eval_20260521_154002\adapt5_60f `
  --gt runs\profiles\tracklet_train_eval_20260521_154002\train_adapt_gt.csv `
  --weights runs\profiles\tracklet_train_eval_20260521_154002\tracklet_mlp_v2_hardtiny_aug.pt `
  --out reports\URAP-UAV\ura6_tracklet_filter_sweep_train_adapt_20260521_v2 `
  --max-frames 60 `
  --score-threshold 0.2 `
  --iou-threshold 0.3
```

## Implementation Note

The first sweep exposed a tracklet grouping bug: `track_id` repeats across sequences, so multi-sequence sweep rows must be grouped by `seq:track_id`, not only `track_id`. Single-sequence infer remains compatible because rows without `seq` still use the original `track_id`.

## Raw Baseline

| Metric | Value |
| --- | ---: |
| GT boxes | 120 |
| drone predictions | 973 |
| TP | 109 |
| FP | 864 |
| FN | 11 |
| precision | 0.1120 |
| recall | 0.9083 |
| frame success rate | 0.9083 |
| FP no-GT frames | 451 |

## Selected Stable Profile

Stable selection target: keep recall within 2 points of raw while reducing FP.

| Setting | Value |
| --- | ---: |
| classifier threshold | 0.95 |
| promotion enabled | 0 |
| promotion score floor | 0.0 |
| promotion max background | 0.0 |
| TP | 109 |
| FP | 615 |
| FN | 11 |
| precision | 0.1506 |
| recall | 0.9083 |
| frame success rate | 0.9083 |
| FP delta | -249 |

Conclusion: this is the best default candidate. It preserves train/adapt recall and removes 249 FP at the operating threshold.

## Selected Hard-Recovery Profile

Hard-recovery target: improve recall by at least 5 points over raw. No swept setting met that target.

| Setting | Value |
| --- | ---: |
| classifier threshold | 0.50 |
| promotion enabled | 1 |
| promotion score floor | 0.30 |
| promotion max background | 0.55 |
| TP | 112 |
| FP | 945 |
| FN | 8 |
| precision | 0.1060 |
| recall | 0.9333 |
| frame success rate | 0.9333 |
| TP delta | +3 |
| FP delta | +81 |

Conclusion: this is not strong enough to become the global default. It can be kept as a diagnostic/hard-case profile, but final selection should be validated on frozen10 before promotion.

## Decision

- Use stable profile as the next default candidate for final frozen10 validation:
  - `--tracklet-classifier-threshold 0.95`
  - `--disable-tracklet-promotion`
- Keep hard-recovery as experimental:
  - `--tracklet-classifier-threshold 0.50`
  - `--tracklet-promotion-score-floor 0.30`
  - `--tracklet-promotion-max-background 0.55`
- Do not select thresholds from frozen10. Frozen10 should only be used in URA-8 final validation.

## Verification

- `pytest tests/test_tracklet_classifier.py -q`: 3 passed
- `python -m qstr_dronedet.cli sweep-tracklet-filter ...`: completed and wrote CSV/JSON summary


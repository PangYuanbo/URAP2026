# URA-10 Temporal-Shape Tracklet Features

Date: 2026-05-21

## Goal

Improve the tracklet classifier representation instead of continuing to loosen hard-recovery gates.

## Added Features

The tracklet dataset now includes temporal-shape and continuity features:

- score, objectness, temporal-drone, background, and final-margin slopes;
- final drone-vs-background margin mean/min/slope;
- background dominance rate and longest streak;
- temporal-over-background rate and longest streak;
- final-score-above-threshold longest streak;
- frame gap max/mean/rate;
- first/last final score and first/last background.

Existing checkpoints remain compatible because inference reads the feature list from the checkpoint. Old checkpoints continue using their original feature set.

## Train/Adapt Run

Commands:

```powershell
python -m qstr_dronedet.cli build-tracklet-dataset `
  --diagnostics <train/adapt hard_recovery diagnostics.jsonl files> `
  --gt-csv runs\profiles\tracklet_train_eval_20260521_154002\train_adapt_gt.csv `
  --out runs\profiles\tracklet_train_eval_20260521_154002\tracklets_v3_temporal_shape `
  --max-frames 60

python -m qstr_dronedet.cli train-tracklet-classifier `
  --csv runs\profiles\tracklet_train_eval_20260521_154002\tracklets_v3_temporal_shape\tracklets.csv `
  --out runs\profiles\tracklet_train_eval_20260521_154002\tracklet_mlp_v3_temporal_shape.pt `
  --epochs 80 `
  --hard-tiny-positive-augments 4
```

Dataset:

```text
num_tracklets: 253
positives: 43
negatives: 210
```

Train-set evaluation reached 1.0 precision/recall, which is treated as overfit and not as a deployment result.

## Frozen10 Validation

Frozen10 was used once for validation only.

| profile | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|
| hard_recovery_raw | 37 | 682 | 17 | 0.0515 | 0.6852 |
| hard_recovery_v3_temporal_shape | 42 | 783 | 12 | 0.0509 | 0.7778 |
| stable | 28 | 257 | 26 | 0.0982 | 0.5185 |

## Interpretation

- The new features preserve recall recovery: TP `37 -> 42`, recall `0.685 -> 0.778`.
- They do not control FP: FP `682 -> 783`.
- Compared with URA-8 hard-recovery (`TP=42`, `FP=752`), v3 keeps recall but adds more FP.
- Therefore the feature implementation is useful infrastructure, but the classifier still needs stronger negatives or a validation-calibrated objective before it can close URA-5.

## Decision

Do not make `tracklet_mlp_v3_temporal_shape.pt` the default checkpoint.

Keep the new features in source because they are backwards-compatible and needed for the next training iteration.

## Artifacts

```text
runs\profiles\tracklet_train_eval_20260521_154002\tracklets_v3_temporal_shape\tracklets.csv
runs\profiles\tracklet_train_eval_20260521_154002\tracklet_mlp_v3_temporal_shape.pt
runs\profiles\frozen10_ura10_v3_temporal_shape_eval_20260521\profile_benchmark_summary.csv
reports\URAP-UAV\ura10_frozen10_v3_temporal_shape_20260521\raw_frame_timeline.png
reports\URAP-UAV\ura10_frozen10_v3_temporal_shape_20260521\filtered_frame_timeline.png
```

## Next Direction

The next fix should improve training data and objective, not gate thresholds:

- mine frozen-like hard negatives only from train/adapt pools;
- add calibration split within train/adapt;
- add penalty for promoted false-positive frames;
- evaluate tracklet classifier by downstream frame metrics, not only tracklet classification accuracy.


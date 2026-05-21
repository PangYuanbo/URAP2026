# Tracklet Classifier Infer Integration

Date: 2026-05-21

Goal:

- connect the trained tracklet classifier to full `infer`;
- use `tracklet_is_drone` to suppress frame-level hard-recovery false positives;
- promote positive hard-tiny tracklets that were previously suppressed by frame-level fusion.

## Implementation

`infer` now accepts:

```text
--tracklet-classifier-weights
--tracklet-classifier-threshold
--tracklet-filter-untracked keep|suppress
--disable-tracklet-promotion
--tracklet-promotion-score-floor
--tracklet-promotion-min-branch-drone
--tracklet-promotion-max-background
```

When `--tracklet-classifier-weights` is provided, `infer` runs the normal frame pipeline first, then performs a post-video tracklet pass:

1. read `diagnostics.jsonl`;
2. group rows by `track_id`;
3. score each tracklet with the `TrackletMLP`;
4. move raw files to:

```text
predictions_raw.jsonl
diagnostics_raw.jsonl
```

5. write filtered standard outputs back to:

```text
predictions.jsonl
diagnostics.jsonl
```

This keeps existing evaluation scripts compatible while preserving raw outputs for audit.

## Filtering Behavior

If a frame row is predicted `drone` but its tracklet has:

```text
tracklet_is_drone = false
```

then the row is rewritten as:

```text
predicted_class = background
final_drone_score = 0.0
diagnostic_cause += tracklet_rejected
```

If a tracklet is classified as drone, the row receives:

```text
tracklet_classifier_prob
tracklet_is_drone
```

For positive tracklets, the post-pass can also promote suppressed frame rows to drone when branch evidence is sufficient:

```text
diagnostic_cause += tracklet_promoted
```

Already-drone rows on positive tracklets receive:

```text
diagnostic_cause += tracklet_confirmed
```

and their score is lifted to at least:

```text
tracklet_promotion_score_floor * P(tracklet_is_drone)
```

## Profile Script Support

The profile runners now pass through the same options:

```text
tools\run_qstr_hard_recovery_profile.ps1
tools\run_qstr_stable_profile.ps1
```

Example:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_hard_recovery_profile.ps1 `
  -Video D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\raw_videos\test\visible\20190925_111757_1_7\visible.mp4 `
  -Out runs\profiles\tracklet_integrated_infer_smoke\1_7_60f_promote_v3 `
  -Device 0 `
  -MaxFrames 60 `
  -AllowTrackerOnlyHardTinyRecovery `
  -TrackletClassifierWeights runs\profiles\tracklet_train_eval_20260521_154002\tracklet_mlp_v2_hardtiny_aug.pt `
  -TrackletClassifierThreshold 0.5
```

## 1_7 Smoke Result

Sequence:

```text
20190925_111757_1_7
first 60 frames
score threshold = 0.20
IoU threshold = 0.30
```

Raw output:

```text
pred_drone=117
TP=0
FP=117
FN=6
precision=0.000
recall=0.000
```

Tracklet-filtered output:

```text
pred_drone=101
TP=2
FP=99
FN=4
precision=0.020
recall=0.333
```

Recovered hits:

| frame | track_id | IoU | score | cause |
|---:|---:|---:|---:|---|
| 10 | 19 | 0.593 | 0.216 | tracklet_promoted |
| 50 | 40 | 0.595 | 0.218 | tracklet_confirmed |

Interpretation:

- the integrated post-pass now changes final `predictions.jsonl`, not only a separate offline benchmark CSV;
- `1_7` is no longer zero-recall in full `infer`;
- false positives are reduced but still high on this hard profile, so the next global evaluation should run frozen10 with tracklet filtering enabled and compare against the previous hard-recovery baseline.

## Validation

```text
pytest tests -q
46 passed
```

Generated runs and model checkpoints remain local-only and should not be committed.

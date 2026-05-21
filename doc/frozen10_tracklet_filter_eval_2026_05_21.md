# Frozen10 Tracklet Filter Evaluation

Date: 2026-05-21

Goal:

- run full frozen10 hard-recovery profile with the integrated tracklet classifier;
- compare the raw hard-recovery outputs against the final `predictions.jsonl` after tracklet filtering and promotion;
- keep the evaluation fixed to the same frozen10, first 60 frames, score threshold `0.20`, IoU threshold `0.30`.

## Command

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_frozen10_profile_benchmark.ps1 `
  -HeldoutRoot D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq `
  -AnnotationsCsv D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\annotations\qstr_real_boxes.csv `
  -OutRoot runs\profiles\frozen10_tracklet_filter_eval_20260521 `
  -Device 0 `
  -MaxVideos 10 `
  -MaxFrames 60 `
  -SkipStable `
  -TrackletClassifierWeights runs\profiles\tracklet_train_eval_20260521_154002\tracklet_mlp_v2_hardtiny_aug.pt `
  -TrackletClassifierThreshold 0.5
```

The run writes both:

- `predictions_raw.jsonl` / `diagnostics_raw.jsonl`
- filtered `predictions.jsonl` / `diagnostics.jsonl`

## Overall Result

| output | GT | pred_drone | TP | FP | FN | precision | recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw hard-recovery | 54 | 719 | 37 | 682 | 17 | 0.051 | 0.685 |
| tracklet filtered/promoted | 54 | 804 | 43 | 761 | 11 | 0.053 | 0.796 |

Interpretation:

- Tracklet integration improved recall substantially: `0.685 -> 0.796`.
- Precision improved only slightly: `0.051 -> 0.053`.
- Absolute false positives increased: `682 -> 761`.
- So the current tracklet filter is useful as a recovery mechanism, but promotion is still too permissive for a low-FP default profile.

## Per-Sequence Result

| sequence | GT | raw TP | raw FP | raw recall | filtered TP | filtered FP | filtered recall | delta TP | delta FP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20190925_111757_1_5 | 6 | 6 | 90 | 1.000 | 5 | 114 | 0.833 | -1 | +24 |
| 20190925_111757_1_6 | 6 | 4 | 58 | 0.667 | 5 | 54 | 0.833 | +1 | -4 |
| 20190925_111757_1_7 | 6 | 0 | 117 | 0.000 | 2 | 99 | 0.333 | +2 | -18 |
| 20190925_111757_1_8 | 0 | 0 | 26 | 0.000 | 0 | 11 | 0.000 | 0 | -15 |
| 20190925_111757_1_9 | 6 | 1 | 4 | 0.167 | 5 | 59 | 0.833 | +4 | +55 |
| 20190925_124000_1_1 | 6 | 6 | 75 | 1.000 | 5 | 75 | 0.833 | -1 | 0 |
| 20190925_124000_1_10 | 6 | 2 | 14 | 0.333 | 4 | 44 | 0.667 | +2 | +30 |
| 20190925_124000_1_2 | 6 | 6 | 99 | 1.000 | 6 | 83 | 1.000 | 0 | -16 |
| 20190925_124000_1_3 | 6 | 6 | 98 | 1.000 | 5 | 132 | 0.833 | -1 | +34 |
| 20190925_124000_1_4 | 6 | 6 | 101 | 1.000 | 6 | 90 | 1.000 | 0 | -11 |

## Conclusion

The integrated tracklet classifier is not just a `1_7` one-off:

- it improves recall on several hard sequences;
- it fixes `1_7` from zero recall to `0.333`;
- it reduces FP on some sequences, including `1_7`, `1_8`, `1_2`, and `1_4`.

But the promotion rule adds too many detections on some sequences:

- `1_9`: `+4 TP`, but `+55 FP`;
- `1_10`: `+2 TP`, but `+30 FP`;
- `1_3`: `-1 TP`, `+34 FP`.

Recommended next step:

- keep tracklet filtering as a hard-recovery/research profile;
- make the default profile use reject-only mode or stricter promotion;
- run a train/adapt threshold sweep for:
  - `tracklet-classifier-threshold`;
  - `tracklet-promotion-score-floor`;
  - `tracklet-promotion-max-background`;
  - disable/enable promotion.

Do not tune these on frozen10.

## Artifacts

```text
runs\profiles\frozen10_tracklet_filter_eval_20260521\profile_benchmark_summary.json
runs\profiles\frozen10_tracklet_filter_eval_20260521\raw_vs_tracklet_filtered_summary.json
```

Generated predictions, diagnostics, and checkpoints remain local-only and should not be committed.

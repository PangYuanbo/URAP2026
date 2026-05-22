# URA-8 Frozen10 Final Validation

Date: 2026-05-21

## Goal

Validate the URA-7 fixed profiles once on frozen10. This run is final validation only; no frozen10 threshold tuning was performed.

## Command

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_frozen10_profile_benchmark.ps1 `
  -HeldoutRoot D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq `
  -AnnotationsCsv D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\annotations\qstr_real_boxes.csv `
  -OutRoot runs\profiles\frozen10_ura8_final_20260521 `
  -Device 0 `
  -MaxVideos 10 `
  -MaxFrames 60
```

Per-frame timelines:

```powershell
python -m qstr_dronedet.cli analyze-frame-failures `
  --run-root runs\profiles\frozen10_ura8_final_20260521 `
  --gt D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\annotations\qstr_real_boxes.csv `
  --out reports\URAP-UAV\ura8_frozen10_final_hard_recovery_20260521 `
  --profile hard_recovery `
  --score-threshold 0.2 `
  --iou-threshold 0.3 `
  --max-frames 60
```

## Overall Result

| profile | prediction file | GT | pred drone | TP | FP | FN | precision | recall | delta TP vs raw | delta FP vs raw | delta recall vs raw |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_recovery_raw | `predictions_raw.jsonl` | 54 | 719 | 37 | 682 | 17 | 0.0515 | 0.6852 | 0 | 0 | 0.0000 |
| hard_recovery | `predictions.jsonl` | 54 | 794 | 42 | 752 | 12 | 0.0529 | 0.7778 | +5 | +70 | +0.0926 |
| stable | `predictions.jsonl` | 54 | 285 | 28 | 257 | 26 | 0.0982 | 0.5185 | -9 | -425 | -0.1667 |

## Per-Frame Failure Summary

Hard-recovery raw:

| metric | value |
|---|---:|
| frame success rate | 0.6852 |
| false_positive_no_gt frames | 393 |
| no_drone_prediction frames | 10 |
| proposal/localization failure frames | 7 |

Hard-recovery filtered/promoted:

| metric | value |
|---|---:|
| frame success rate | 0.7778 |
| false_positive_no_gt frames | 480 |
| no_drone_prediction frames | 4 |
| proposal/localization failure frames | 7 |
| weak localization frames | 1 |

Stable filtered:

| metric | value |
|---|---:|
| frame success rate | 0.5185 |
| false_positive_no_gt frames | 241 |
| no_drone_prediction frames | 21 |
| proposal/localization failure frames | 5 |

## Interpretation

- The hard-recovery profile is useful for recall: `37 -> 42` TP and `0.685 -> 0.778` recall.
- The hard-recovery profile still increases false positives: `682 -> 752` FP and `393 -> 480` false-positive no-GT frames.
- The stable profile controls FP strongly: `682 -> 257` FP versus raw hard-recovery, but it loses too much recall: `0.685 -> 0.519`.
- Therefore URA-5 should stay open. The current system has a valid low-FP profile and a valid recall-recovery profile, but not a single validated profile that simultaneously controls FP and improves recall over raw hard-recovery.

## Decision

- Do not make hard-recovery the default pipeline.
- Use stable only as the conservative low-FP operating mode.
- Keep hard-recovery as a diagnostic/research profile for hard tiny and fallback cases.
- Next work should combine stable rejection with selective hard-recovery only on frames or tracklets where recovery evidence is stronger than background/artifact evidence.

## Artifacts

```text
runs\profiles\frozen10_ura8_final_20260521\profile_benchmark_summary.json
runs\profiles\frozen10_ura8_final_20260521\profile_benchmark_summary.csv
reports\URAP-UAV\ura8_frozen10_final_hard_recovery_20260521\raw_frame_timeline.png
reports\URAP-UAV\ura8_frozen10_final_hard_recovery_20260521\filtered_frame_timeline.png
reports\URAP-UAV\ura8_frozen10_final_hard_recovery_20260521\per_frame_raw.csv
reports\URAP-UAV\ura8_frozen10_final_hard_recovery_20260521\per_frame_filtered.csv
reports\URAP-UAV\ura8_frozen10_final_stable_20260521\raw_frame_timeline.png
reports\URAP-UAV\ura8_frozen10_final_stable_20260521\filtered_frame_timeline.png
reports\URAP-UAV\ura8_frozen10_final_stable_20260521\per_frame_raw.csv
reports\URAP-UAV\ura8_frozen10_final_stable_20260521\per_frame_filtered.csv
```

## Recommended Next Issue

Create a selective profile that only enables hard-recovery promotion when sequence/frame evidence indicates a recoverable tiny target:

- primary detector weak or missing;
- fallback/tracklet support exists across frames;
- temporal evidence is consistently stronger than crop/background;
- background/artifact evidence stays below a calibrated limit;
- per-sequence FP budget is enforced.


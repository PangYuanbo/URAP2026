# URA-15 Expanded Non-Frozen Hard-Recovery Profiles

## Goal

Run QSTR hard-recovery on more non-frozen Anti-UAV sequences, then rebuild proposal-tracklet data and rerun train/adapt-only model selection. This follows the URA-14 conclusion: do not tune frozen10 thresholds; expand the non-frozen proposal-tracklet data first.

## Implementation

New batch script:

- `tools/run_qstr_nonfrozen_hard_recovery_batch.ps1`

It reads a QSTR recording manifest, skips named sequences, runs `run_qstr_hard_recovery_profile.ps1` per video, and writes `batch_summary.csv`.

Example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& tools\run_qstr_nonfrozen_hard_recovery_batch.ps1 `
  -Manifest 'D:\datasets\Anti-UAV300\qstr_train_visible_20seq\annotations\recording_manifest.csv' `
  -OutRoot 'runs\profiles\expanded_nonfrozen_ura15\train20_extra5_60f' `
  -MaxSequences 5 `
  -MaxFrames 60 `
  -SkipExisting `
  -ExcludeSequences @('20190925_101846_1_1','20190925_101846_1_2','20190925_101846_1_3','20190925_101846_1_6','20190925_101846_1_7') `
  -Device 0"
```

## New Non-Frozen Profile Runs

Added 5 non-frozen train-visible sequences beyond the existing train5:

- `20190925_101846_1_8` (`tiny`)
- `20190925_130434_1_3` (`fast_target`)
- `20190925_130434_1_4` (`fast_target`)
- `20190925_130434_1_7` (`fast_target`)
- `20190925_130434_1_9` (`fast_target`)

These were written locally under:

- `runs\profiles\expanded_nonfrozen_ura15\train20_extra5_60f`

## Expanded Tracklet Data

Merged GT:

- `runs\profiles\expanded_nonfrozen_ura15\train20_adapt5_gt.csv`
- rows: 2000

Expanded proposal-tracklet dataset using train5 + extra5 + adapt5:

- sequences: 15
- proposal tracklets: 553
- positives: 68
- negatives: 485
- positive: 66
- hard tiny positive: 2
- high-score detector FP: 335
- easy background: 150

Expanded original tracker tracklets:

- tracklets: 150
- positives: 14
- negatives: 136

Mixed expanded tracklet JSONL:

- total: 703
- positives: 82
- negatives: 621

## Train/Adapt Selection

Calibration split:

- train: 10 train-visible sequences
- calibration: adapt5 `20190925_124000_*`

Raw calibration:

- TP: 49
- FP: 297
- FN: 11
- Precision: 0.1416
- Recall: 0.8167

Selected FP-control model:

- max_len: `16`
- hard-positive augments: `0`
- hard-negative augments: `0`
- threshold: `0.7`
- promotion: disabled
- TP: 45
- FP: 216
- FN: 15
- Precision: 0.1724
- Recall: 0.7500

Best high-recall candidates reached recall `0.8000`, but increased FP above raw. Example:

- TP: 48
- FP: 312
- Recall: 0.8000
- FP delta: +15

## Decision

No deployable model was selected. The expanded train10/adapt5 data improves FP-control capacity, but all configurations either:

- reduce FP while losing too much recall, or
- preserve recall only by increasing FP.

Because the model did not pass train/adapt calibration, frozen10 was not evaluated in this step.

## Next Step

Continue expanding non-frozen coverage before frozen10 validation:

- run the remaining `qstr_train_visible_20seq` sequences, especially static-hovering and tiny cases;
- optionally add `qstr_train_visible_val_5seq`;
- rebuild proposal-tracklets;
- rerun train/adapt selection;
- only run frozen10 once a candidate passes calibration.

## Verification

```text
pytest tests -q
55 passed

python -m qstr_dronedet.cli --help
passed
```

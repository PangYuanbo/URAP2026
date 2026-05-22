# URA-14 Hard-Positive Proposal Tracklet Expansion

## Goal

Continue from URA-13 by expanding the non-frozen proposal-tracklet training data, with emphasis on hard positive proposal tracks. The goal is to reduce false positives without over-suppressing true tiny drone tracks.

Frozen10 was not used for threshold selection.

## Implementation

New training support:

- `train_tracklet_sequence_classifier(..., hard_positive_augments=...)`
- `select-tracklet-sequence-model --hard-positive-augments-values ...`

Hard-positive augmentation identifies positive tracklets that are likely to be difficult:

- short tracklets;
- low objectness;
- fallback/weak-detector source;
- weak detector plus temporal evidence;
- small mean box side;
- explicit `hard_*` bucket.

For these positives, the trainer creates degraded positive copies:

- objectness/final score are reduced;
- crop/feature/final drone evidence is capped;
- temporal evidence remains stronger than crop/background;
- background evidence is increased moderately;
- source is tagged with `hard_positive_aug`.

New reproducibility support:

- `merge-tracklet-jsonl`

This formalizes the previous manual mixed dataset step by combining original tracker-derived tracklets with proposal-derived tracklets and tagging `meta.dataset_source`.

## Experiments

Dataset:

- original tracker tracklets: 253
- proposal tracklets: 351
- mixed total: 604
- proposal high-score detector FP: 201

Selection command swept:

- hard-positive augments: `0`, `2`, `4`
- hard-negative augments: `0`, `1`
- max length: `8`, `16`
- thresholds: `0.5`, `0.7`, `0.85`
- promotion: off/on

## Train/Adapt Calibration Result

Raw calibration:

- TP: 49
- FP: 297
- FN: 11
- Precision: 0.1416
- Recall: 0.8167

Selected configuration:

- max length: `8`
- hard-positive augments: `0`
- hard-negative augments: `0`
- threshold: `0.7`
- promotion: disabled

Selected calibration metrics:

- TP: 49
- FP: 255
- FN: 11
- Precision: 0.1612
- Recall: 0.8167

The selector did not choose hard-positive augmentation because the non-augmented mixed model already preserved recall on the adapt split while reducing FP by 42.

## Frozen10 One-Time Validation

Raw frozen10:

- TP: 37
- FP: 682
- FN: 17
- Precision: 0.0515
- Recall: 0.6852

Selected hard-positive-sweep model:

- TP: 34
- FP: 735
- FN: 20
- Precision: 0.0442
- Recall: 0.6296

Delta:

- TP: -3
- FP: +53
- Recall: -0.0556
- Precision: -0.0072

## Decision

This did not solve URA-5. The code path is useful, but the current train/adapt pool is too narrow and the selected model still transfers poorly to frozen10.

Do not make this model default.

## Next Step

Run QSTR hard-recovery profiles on more non-frozen Anti-UAV sequences, then rebuild proposal tracklets with the new builder. The missing piece is not another threshold on the current 10 sequences; it is broader non-frozen proposal-tracklet coverage, especially hard positives that resemble frozen10.

Recommended next issue:

- export/run an expanded non-frozen profile set from `qstr_train_visible_20seq` and `qstr_train_visible_val_5seq`;
- build proposal tracklets from those roots;
- keep frozen10 untouched until train/adapt selection picks a deployable model.

## Verification

```text
pytest tests -q
54 passed

python -m qstr_dronedet.cli select-tracklet-sequence-model --help
passed
```

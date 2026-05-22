# URA-16 Full Non-Frozen Profile Expansion

## Goal

Continue URA-15 by running the remaining train-visible Anti-UAV sequences and adding `train_visible_val_5seq`, then rebuild proposal-tracklet data and rerun train/adapt-only selection.

Frozen10 remains untouched unless train/adapt calibration selects a deployable candidate.

## Profile Runs

New train20 remaining sequences:

- `20190925_131530_1_1` fast_target
- `20190925_131530_1_2` fast_target
- `20190925_131530_1_4` static_hovering
- `20190925_131530_1_5` static_hovering
- `20190925_131530_1_6` static_hovering
- `20190925_131530_1_7` static_hovering
- `20190925_133630_1_1` fast_target
- `20190925_133630_1_10` tiny
- `20190925_133630_1_2` fast_target
- `20190925_133630_1_3` static_hovering

New train-visible-val sequences:

- `20190925_133630_1_4` fast_target
- `20190925_133630_1_5` tiny
- `20190925_133630_1_6` tiny
- `20190925_133630_1_7` tiny
- `20190925_133630_1_8` static_hovering

Local output roots:

- `runs\profiles\expanded_nonfrozen_ura16\train20_remaining10_60f`
- `runs\profiles\expanded_nonfrozen_ura16\trainval5_60f`

## Expanded Data

Merged GT:

- `runs\profiles\expanded_nonfrozen_ura16\train20_trainval5_adapt5_gt.csv`
- rows: 2400

Proposal-tracklet dataset:

- sequences: 30
- proposal tracklets: 1299
- positives: 122
- negatives: 1177
- positive bucket: 119
- hard tiny positive bucket: 3
- high-score detector FP: 798
- easy background: 379

Original tracker-tracklet dataset:

- tracklets: 666
- positives: 55
- negatives: 611

Mixed original + proposal JSONL:

- total tracklets: 1965
- positives: 177
- negatives: 1788

## Train/Adapt Selection

Calibration split:

- train: 25 train/train-val visible sequences
- calibration: adapt5 `20190925_124000_*`

Raw calibration:

- TP: 49
- FP: 297
- FN: 11
- precision: 0.1416
- recall: 0.8167

Selected FP-control candidate:

- max_len: 16
- hard-positive augments: 0
- hard-negative augments: 0
- threshold: 0.7
- promotion: disabled
- TP: 45
- FP: 220
- FN: 15
- precision: 0.1698
- recall: 0.7500

Best high-recall candidates:

- reached recall: 0.8000
- but increased FP above raw, e.g. TP 48 / FP 326

## Decision

No candidate passed train/adapt calibration:

- FP-control candidates reduce FP but lose too much recall.
- High-recall candidates preserve most recall only by increasing FP beyond raw.

Frozen10 was not evaluated in this step.

## Interpretation

The larger non-frozen data pool helps expose more hard negatives, but the model still learns a suppression rule that is too blunt. The bottleneck now looks less like "not enough random non-frozen data" and more like missing hard positive coverage that resembles the adapt/frozen failure cases.

## Next Step

Before another frozen10 validation:

- mine hard positive proposal tracklets specifically from low-recall train/adapt frames;
- include candidate-level positives that failed final scoring, not only final diagnostics;
- consider lowering the calibration objective from strict raw-recall preservation to two-profile selection:
  - stable FP-control profile;
  - hard-recovery recall profile;
- keep frozen10 untouched until a profile passes its intended train/adapt criterion.

## Verification

The profile/data run used existing source. The latest source validation from this cycle remains:

```text
pytest tests -q
55 passed

python -m qstr_dronedet.cli --help
passed
```

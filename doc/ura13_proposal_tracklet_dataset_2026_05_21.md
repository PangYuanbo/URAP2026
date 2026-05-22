# URA-13 Proposal-Tracklet Dataset

## Goal

Build a more realistic non-frozen proposal-tracklet dataset for the sequence-level tracklet classifier. The previous dataset used tracker-derived diagnostic tracklets. URA-13 adds a data builder that reconstructs proposal tracklets directly from per-frame diagnostics/proposals, including short detector/fallback tracks and hard false positives.

Frozen10 remains final validation only. No frozen10 threshold tuning was performed.

## Implementation

New CLI:

```powershell
python -m qstr_dronedet.cli build-proposal-tracklet-dataset `
  --run-roots runs\profiles\tracklet_train_eval_20260521_154002\train5_60f runs\profiles\tracklet_train_eval_20260521_154002\adapt5_60f `
  --gt-csv runs\profiles\tracklet_train_eval_20260521_154002\train_adapt_gt.csv `
  --out runs\profiles\tracklet_train_eval_20260521_154002\proposal_tracklets_ura13 `
  --profile hard_recovery `
  --diagnostics-name diagnostics_raw.jsonl `
  --max-frames 60
```

New source:

- `qstr_dronedet/tracking/proposal_tracklets.py`

The builder:

- reads `diagnostics_raw.jsonl` where available;
- greedily relinks candidate rows by frame gap, center distance, and IoU;
- does not require existing tracker `track_id`;
- labels each proposal tracklet against GT;
- writes both CSV aggregate features and JSONL row sequences;
- assigns buckets:
  - `positive`
  - `hard_tiny_positive`
  - `high_score_detector_fp`
  - `motion_alignment_artifact`
  - `easy_background`

## Dataset Result

Using current train/adapt 10-sequence pool:

- total proposal tracklets: 351
- positives: 43
- negatives: 308
- high-score detector FP: 201
- easy background: 107

This is more realistic than the tracker-only dataset because the negative side now includes many detector-like false proposal tracks.

## Sequence Model Selection

Two data variants were tested.

### Proposal-only

Calibration result:

- raw: TP 49, FP 297, recall 0.8167, precision 0.1416
- selected proposal-only GRU: TP 25, FP 115, recall 0.4167, precision 0.1786

Decision:

- too conservative;
- not deployable.

### Mixed Original + Proposal

The original tracker-tracklet JSONL was mixed with proposal-tracklet JSONL:

- original tracker tracklets: 253
- proposal tracklets: 351
- total: 604
- proposal high-score detector FP: 201
- proposal positives: 43

Calibration result:

- raw: TP 49, FP 297, recall 0.8167, precision 0.1416
- selected mixed GRU: TP 48, FP 252, recall 0.8000, precision 0.1600

This passed the train/adapt calibration rule.

## Frozen10 One-Time Validation

The selected mixed-data GRU was evaluated once on frozen10.

Raw frozen10:

- TP: 37
- FP: 682
- FN: 17
- Precision: 0.0515
- Recall: 0.6852
- False-positive no-GT frames: 393

Mixed proposal sequence GRU:

- TP: 25
- FP: 562
- FN: 29
- Precision: 0.0426
- Recall: 0.4630
- False-positive no-GT frames: 312

Delta:

- TP: -12
- FP: -120
- Recall: -0.2222
- Precision: -0.0089
- False-positive no-GT frames: -81

## Conclusion

The proposal-tracklet builder is useful and should stay. It exposes realistic high-score detector false positives and gives the sequence classifier harder negatives.

But the current train/adapt proposal data is still not enough to generalize. The selected model becomes too conservative on frozen10 and suppresses too many true positives.

Current decision:

- Do not use the mixed proposal sequence model as default.
- Keep URA-5 open.
- Use the proposal-tracklet builder as the data pipeline for larger non-frozen training.

## Next Step

Expand the non-frozen proposal-tracklet dataset before another model-selection run:

- include more Anti-UAV train/test-like sequences outside frozen10;
- explicitly oversample hard positive proposal tracklets with short length, low objectness, and tiny boxes;
- keep high-score detector FP buckets;
- select on train/adapt calibration;
- run frozen10 once after selection.

## Verification

```text
pytest tests -q
54 passed

python -m qstr_dronedet.cli --help
passed
```

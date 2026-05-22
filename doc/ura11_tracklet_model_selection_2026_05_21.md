# URA-11 Tracklet Model Selection With Calibration Split

## Goal

URA-11 moves tracklet-classifier selection away from train tracklet accuracy. The new path trains candidate classifiers on train sequences only, reserves adapt sequences for calibration, and selects by downstream frame metrics: TP, FP, FN, precision, recall, and frame success rate.

Frozen10 remains a final validation set only. No frozen10 thresholds are swept or selected.

## Implementation

New CLI:

```powershell
python -m qstr_dronedet.cli select-tracklet-model `
  --tracklets runs\profiles\tracklet_train_eval_20260521_154002\tracklets_v3_temporal_shape\tracklets.csv `
  --run-roots runs\profiles\tracklet_train_eval_20260521_154002\train5_60f runs\profiles\tracklet_train_eval_20260521_154002\adapt5_60f `
  --gt runs\profiles\tracklet_train_eval_20260521_154002\train_adapt_gt.csv `
  --out runs\profiles\tracklet_train_eval_20260521_154002\tracklet_model_selection_ura11_v2 `
  --profile hard_recovery `
  --calib-seq-patterns 20190925_124000_* `
  --epochs-values 20 40 `
  --hard-tiny-positive-augments-values 0 2 `
  --hard-negative-augments-values 0 2 `
  --classifier-thresholds 0.5 0.7 0.85 `
  --promotion-enabled-values 0 1 `
  --selective-promotion `
  --max-frames 60
```

Sequence split:

- Train: `20190925_101846_1_1`, `1_2`, `1_3`, `1_6`, `1_7`
- Calibration: `20190925_124000_1_5`, `1_6`, `1_7`, `1_8`, `1_9`

The tool writes:

- `train_tracklets_hn*.csv`
- `calibration_tracklets.csv`
- `model_selection.csv`
- `model_selection_summary.json`
- `selected_tracklet_classifier.pt`

## Calibration Result

Raw calibration baseline:

- TP: 49
- FP: 297
- FN: 11
- Precision: 0.1416
- Recall: 0.8167

Selected high-recall profile:

- candidate: `tracklet_v3_candidate_002.pt`
- classifier threshold: `0.50`
- hard tiny positive augments: `2`
- hard negative augments: `0`
- selective promotion: enabled
- per-sequence promotion budget: `1`
- TP: 48
- FP: 288
- FN: 12
- Precision: 0.1429
- Recall: 0.8000

This passes the calibration rule because recall is within the allowed 0.02 drop from raw while FP and precision are slightly better.

Best FP-control calibration profile:

- threshold: `0.85`
- promotion: disabled
- TP: 27
- FP: 121
- FN: 33
- Precision: 0.1824
- Recall: 0.4500

This reduces FP on calibration, but the recall drop is too large for hard-recovery.

## Frozen10 Final Validation

The selected high-recall profile was then evaluated once on frozen10. Frozen10 was not used for threshold selection.

Raw frozen10 hard-recovery baseline:

- TP: 37
- FP: 682
- FN: 17
- Precision: 0.0515
- Recall: 0.6852
- Frame success rate: 0.6852

Selected high-recall frozen10 result:

- TP: 36
- FP: 703
- FN: 18
- Precision: 0.0487
- Recall: 0.6667
- Frame success rate: 0.6667

Delta:

- TP: -1
- FP: +21
- Recall: -0.0185
- Precision: -0.0027

The selected model does not generalize to frozen10 and should not replace the default hard-recovery profile.

The FP-control profile was also evaluated once as a diagnostic:

- TP: 28
- FP: 543
- FN: 26
- Precision: 0.0490
- Recall: 0.5185

It reduces FP by 139, but loses 9 TP and still does not improve precision over raw. It is not a useful default.

## Conclusion

URA-11 confirms that the previous v3 tracklet features and simple MLP are not enough. Calibration-split model selection prevents us from promoting a train-accuracy-overfit classifier, and frozen10 shows the model still fails domain transfer.

Current decision:

- Do not make the v3 tracklet classifier the default hard-recovery filter.
- Keep `URA-5` open.
- Keep `URA-9` open or continue it with stronger tracklet representation work.

Recommended next issue:

- Replace the current small MLP-only tracklet classifier with a real sequence-level model or richer proposal-tracklet dataset:
  - train on more non-frozen Anti-UAV sequences;
  - include true detector proposal tracklets, not only derived diagnostic tracklets;
  - add sequence ordering directly with a GRU/TCN over per-frame evidence;
  - evaluate by train/adapt calibration frame metrics before one frozen10 validation.

## Verification

Local source validation:

```text
pytest tests -q
51 passed

python -m qstr_dronedet.cli --help
passed
```

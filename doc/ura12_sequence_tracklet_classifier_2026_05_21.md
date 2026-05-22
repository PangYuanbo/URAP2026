# URA-12 Sequence-Level Tracklet Classifier

## Goal

URA-12 replaces the aggregate-only tracklet MLP with a sequence-level classifier that reads per-frame evidence inside a tracklet. The goal is to let the model learn temporal shape directly:

- crop/feature/temporal/final drone evidence over time;
- background and artifact pressure over time;
- score trends, frame gaps, center motion, and source flags;
- fallback/tracker/yolo/motion source continuity.

This is meant to address the URA-11 finding: train tracklet accuracy and aggregate features were not enough for frozen10 generalization.

## Implementation

New module:

- `qstr_dronedet/tracking/tracklet_sequence_classifier.py`

New CLI:

```powershell
python -m qstr_dronedet.cli select-tracklet-sequence-model `
  --tracklets-jsonl runs\profiles\tracklet_train_eval_20260521_154002\tracklets_v3_temporal_shape\tracklets.jsonl `
  --run-roots runs\profiles\tracklet_train_eval_20260521_154002\train5_60f runs\profiles\tracklet_train_eval_20260521_154002\adapt5_60f `
  --gt runs\profiles\tracklet_train_eval_20260521_154002\train_adapt_gt.csv `
  --out runs\profiles\tracklet_train_eval_20260521_154002\tracklet_sequence_model_selection_ura12 `
  --profile hard_recovery `
  --calib-seq-patterns 20190925_124000_* `
  --epochs-values 20 `
  --max-len-values 8 16 `
  --hard-negative-augments-values 0 2 `
  --classifier-thresholds 0.5 0.7 0.85 `
  --promotion-enabled-values 0 1 `
  --selective-promotion `
  --max-frames 60
```

Model:

- GRU over per-frame features.
- Padded/truncated sequence length.
- Weighted cross entropy for positive/negative imbalance.
- Hard-negative tracklet oversampling option.

Frame features include:

- objectness and final drone score;
- crop/feature/temporal/final drone probabilities;
- crop/temporal/final background probabilities;
- alignment-artifact probability;
- motion/alignment/tracker metadata;
- box side, center step, frame gap;
- source flags for yolo/fallback/tracker/motion;
- branch deltas such as temporal minus crop and temporal minus background.

## Train/Adapt Calibration Result

Frozen10 was not used for model selection.

Split:

- Train: `20190925_101846_1_1`, `1_2`, `1_3`, `1_6`, `1_7`
- Calibration: `20190925_124000_1_5`, `1_6`, `1_7`, `1_8`, `1_9`

Raw calibration baseline:

- TP: 49
- FP: 297
- FN: 11
- Precision: 0.1416
- Recall: 0.8167

Selected sequence-GRU calibration config:

- checkpoint: `tracklet_sequence_candidate_003.pt`
- max length: `16`
- threshold: `0.50`
- promotion: disabled
- hard negative augments: `0`
- TP: 49
- FP: 262
- FN: 11
- Precision: 0.1576
- Recall: 0.8167

Calibration interpretation:

- Recall preserved exactly.
- FP reduced by 35.
- Precision improved from 0.1416 to 0.1576.
- This is better than URA-11's aggregate MLP calibration result.

## Frozen10 One-Time Validation

Frozen10 was evaluated once after selecting the config from train/adapt.

Raw frozen10 hard-recovery baseline:

- TP: 37
- FP: 682
- FN: 17
- Precision: 0.0515
- Recall: 0.6852
- Frame success rate: 0.6852
- False-positive no-GT frames: 393

Selected sequence-GRU frozen10 result:

- TP: 36
- FP: 650
- FN: 18
- Precision: 0.0525
- Recall: 0.6667
- Frame success rate: 0.6667
- False-positive no-GT frames: 380

Delta:

- TP: -1
- FP: -32
- Recall: -0.0185
- Precision: +0.0010
- False-positive no-GT frames: -13

## Decision

The sequence-GRU is directionally better than the aggregate MLP:

- URA-11 selected MLP on frozen10: TP 36, FP 703, recall 0.6667.
- URA-12 selected GRU on frozen10: TP 36, FP 650, recall 0.6667.

So the sequence model reduces FP by 53 versus URA-11's selected MLP result at the same TP/FN.

However, it still loses one TP relative to raw hard-recovery. It should not become the default profile yet.

## Next Step

The next fix should improve the data, not tune frozen10:

- generate more non-frozen proposal tracklets from train/adapt pool;
- include detector proposal tracklets before final fusion, not only diagnostics after fusion;
- label hard positive short/low-objectness tracklets and high-score FP tracklets;
- retrain the sequence model on that richer proposal-tracklet dataset;
- select again on train/adapt calibration;
- run frozen10 once after selection.

## Verification

```text
pytest tests -q
53 passed

python -m qstr_dronedet.cli --help
passed
```

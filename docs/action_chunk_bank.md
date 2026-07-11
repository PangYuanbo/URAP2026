# Action Chunk Bank

This repository uses an original temporal candidate-ranking design. The deployable path does not use third-party tracker memory, masks, object pointers, or tracker proposal logic.

## Runtime inputs

- Current detector candidates.
- Background camera homography estimated from optical flow.
- Past Action Chunks only.
- Real sequence FPS and elapsed timestamps. The current NPS manifest contains 29.97 FPS clips and 59.94 FPS clips.

The deployable scorer has no backward-file argument. Reversed candidate passes are training-only supervision or separately reported offline experiments.

## Memory layout

- Short bank: the latest 1 second, sampled into 8 dense action tokens.
- Long bank: the latest 3 seconds, compressed into 16 motion-pattern tokens.
- Track persistence is measured in seconds, not a fixed number of frames.
- Velocity and acceleration divide camera-compensated displacement by the real elapsed time.

Short tokens retain residual displacement, velocity, acceleration, scale change, motion IoU, detector confidence, and compatibility. Long tokens summarize average velocity, velocity trend, dominant direction, turning behavior, scale trend, stability, age, and reliable hypotheses.

## Camera compensation

The bank subtracts frame-level camera motion before evaluating candidate motion. Direct box NMS is never applied across different frame coordinate systems. Unmatched reliable tracks remain in the bank and decay according to real age over the 3-second horizon.

## Training and deployment boundary

- Deployment input: raw candidate geometry, immediate past bank, persistent-bank delta, and past-only 0.25/1/3-second neighbor evidence.
- Future or reversed Action Chunk evidence may change training sample weights only.
- Future evidence is never part of the deployed feature vector.
- Test-time runners must not accept or read a backward/future feature file.
- Ground-truth labels are used only to form training targets and calculate offline metrics.
- Offline bidirectional results are reported separately and must not be described as causal or real-time.

## Result interpretation

- Original detector: 93.8417% mAP@0.5.
- Best offline bidirectional Action Chunk result so far: 95.3292% mAP@0.5.
- Best completed strict causal result: 94.8890% mAP@0.5 (V60 bounded residual).
- The candidate-set oracle is 99.2083% mAP@0.5, so the remaining gap is mainly candidate ranking rather than missing boxes.
- V59 direct dual-memory classification was rejected after severe domain shift (95.47% validation versus 86.20% fixed test).
- V61 is regenerating the persistent bank with cross-frame hypotheses projected into the current camera coordinate system before deduplication.

## Ownership boundary

The main implementation lives in:

- `qstr_dronedet/tracking/action_chunk_bank.py`
- `qstr_dronedet/action_chunk_camera_motion.py`
- `tools/score_predictionsgt_action_chunk_bank.py`
- `tools/train_action_chunk_causal_memory.py`

Legacy online-bank imports are compatibility aliases only. New experiments must use the Action Chunk modules and `action_chunk_*` output fields.

# SAMURAI-Style Motion-Aware Memory Direction

Date: 2026-06-23

## Why This Becomes The New Main Direction

SAMURAI is relevant to our tiny-UAV problem because its main contribution is not
"SAM 2 segmentation" by itself. The useful idea is:

```text
do not trust the newest observation blindly
  -> predict object motion
  -> score whether the current observation agrees with motion
  -> write only high-quality observations into memory
  -> use clean memory to survive occlusion, distractors, and re-acquisition
```

This matches our failure mode. A tiny UAV is often only a few pixels, so a single
frame cannot reliably say whether the blob is a UAV, bird, compression artifact,
or detector noise. The identity is carried by consistent video motion and by
clean historical evidence.

Primary paper/source anchors:

- SAMURAI paper: https://arxiv.org/abs/2411.11922
- Official implementation: https://github.com/yangchris11/samurai

Important verified details:

- The paper was submitted on 2024-11-18 and revised on 2024-11-30.
- It adapts SAM 2 for zero-shot visual object tracking with motion-aware memory.
- It addresses SAM 2's fixed-window memory error propagation by using temporal
  motion cues and selective memory updates.
- It reports real-time zero-shot tracking, no retraining/fine-tuning, with gains
  including +7.1 AUC on LaSOT_ext and +3.5 AO on GOT-10k.

## What We Copy, And What We Do Not Copy

We should copy:

- Motion-aware memory selection.
- Kalman/constant-velocity prediction as a cheap 2D state-space prior.
- A hybrid score that combines detector confidence, motion consistency, and
  support evidence before a frame is allowed to update long-term memory.
- Re-acquisition through clean historical memory, not through contaminated recent
  frames.

We should not blindly copy:

- Full SAM 2 segmentation as the first production path. Tiny UAV masks may be
  unstable and the model is heavier than the current onboard direction.
- The VOT assumption that the first frame ground-truth box is always provided.
  Our task is detection-first; initialization must come from low-threshold
  detector evidence or a native video detector.

## Updated Runtime Architecture

The user-proposed structure is directionally correct. The adjusted version is:

```text
video / frame folder
        |
        v
E104 v2 detector or low-threshold YOLO detector
YOLO top-K bbox candidates
        |
        v
candidate selection layer
        |
        +-- default profile:
        |     SAMURAI-style selective motion-memory rescoring
        |     detector score + motion score + support score
        |
        +-- dji-tiny profile:
              motion-memory guided zoom-in re-detection
              crop detector -> remap -> merge -> rescore
        |
        v
hard-reset bbox correction / stale-memory guard
        |
        v
gray NCC tracker support + diagnostics
        |
        v
trajectory.csv / trajectory.json / annotated.mp4
```

One adjustment: gray NCC should not only be a final diagnostic. It is a weak
support proposal/evidence source. It can help bridge short detector misses, but
it must pass the same memory-quality gate before being written into clean memory.

## How This Differs From Our Old Action-Chunk Line

The old action-chunk line asked:

```text
given past boxes, predict future boxes/actions
```

That is useful, but it is too fragile if used as the main detector. A one-pixel
localization error on a tiny target can look like a large action error.

The SAMURAI-style line asks a better detection question:

```text
does this current weak candidate agree with clean memory and plausible motion?
```

So the action/dynamics model moves into the correct role:

- MVP: constant-velocity/Kalman-style motion score.
- Next: learned residual/action-chunk likelihood as an extra scorer.
- Later: memory/action tokens inside the native video detector.

Action chunk remains valuable, but as a motion prior or likelihood term, not as
the only output head.

## Current Code Mapping

The first implementation path is already close to this structure:

- `qstr_dronedet/pipelines/temporal_recovery.py`
  - low-threshold top-K candidates
  - motion-memory rescoring
  - NCC support proposals
  - zoom-in re-detection for the `dji-tiny` profile
  - hard-reset stale-memory correction
  - new SAMURAI-style selective memory write gate
- `tools/run_temporal_recovery_pipeline.py`
  - runs the detector-first temporal recovery pipeline
  - writes `trajectory.csv`, `trajectory.json`, candidate JSONL, optional labels,
    and optional `annotated.mp4`

The 2026-06-23 code update adds explicit memory diagnostics:

```text
memory_quality
memory_write
memory_write_reason
memory_bank_size
emit_detection
emit_reason
```

This makes the next ablations measurable instead of subjective.

The runtime also gates support-only outputs. A gray-NCC candidate can still be
kept as support evidence, but if no detector/zoom evidence is present in the
same frame, it must pass a stricter support-only output gate before it can be
emitted as a final detection. This prevents the old failure mode where NCC drift
became the trajectory.

Matched-FP comparison entry point:

```text
tools/compare_yolo_labels_matched_fp.py
```

Use it after exporting baseline and SAMURAI-style temporal recovery outputs into
YOLO prediction labels. The script sweeps thresholds, derives a false-positive
budget from the selected baseline point, and then selects the highest-recall
candidate setting whose FP count does not exceed that budget.

For full validation runs, use the detached wrappers:

```text
tools/start_compare_yolo_labels_matched_fp_detached.ps1
tools/monitor_compare_yolo_labels_matched_fp.ps1
```

This keeps the run auditable with PID files and logs instead of relying on an
interactive terminal session.

## Current Evidence: Why Selective Memory Is Necessary

Existing ARD100 YOLOMG validation outputs show that a naive NCC/memory support
branch is not enough. On the full `U:\URAP_datasets\ARD100_YOLOMG\val.txt`
split, the matched-FP comparison between the old no-NCC temporal recovery output
and the old NCC-enabled output gives:

| method | selected threshold | TP | FP | FN | precision | recall | F1 | FPPI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `no_ncc` baseline | 0.001 | 18269 | 2493 | 2493 | 0.8799 | 0.8799 | 0.8799 | 0.1201 |
| `old_ncc` under same FP budget | 0.7 | 11271 | 172 | 9491 | 0.9850 | 0.5429 | 0.7000 | 0.0083 |

Result path:

```text
U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\matched_fp_no_ncc_vs_old_ncc\compare.json
```

Interpretation:

- The old NCC path reduces false positives only by losing too much recall.
- This does not disprove motion memory. It shows that support proposals and
  memory writes must be selective.
- The new SAMURAI-style gate therefore controls both memory writes and
  support-only final outputs.
- The next full validation target is the new SAMURAI-style memory gate, not the
  old unconditional NCC behavior.

New full-run entry point for that validation:

```text
tools/start_samurai_memory_gate_yolomg_val_detached.ps1
tools/monitor_samurai_memory_gate_yolomg_val.ps1
```

This writes to a separate output root:

```text
U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\samurai_memory_gate_yolomg_val_full
```

After that run finishes, export/evaluate its labels and run the matched-FP
comparison against `no_ncc`. Only that result can support a claim that the new
SAMURAI-style gate beats the current baseline.

Validation chain:

```text
1. tools/start_samurai_memory_gate_yolomg_val_detached.ps1
   tools/monitor_samurai_memory_gate_yolomg_val.ps1

2. tools/start_samurai_memory_gate_yolomg_val_eval_detached.ps1
   tools/monitor_samurai_memory_gate_yolomg_val_eval.ps1

3. tools/start_samurai_memory_gate_matched_fp_compare_detached.ps1
   tools/monitor_samurai_memory_gate_matched_fp_compare.ps1
```

Claim rule:

```text
samurai_gate recall must exceed no_ncc recall under the same FP budget
```

Smoke evidence after adding the support-only output gate:

```text
split: first 2000 frames of U:\URAP_datasets\ARD100_YOLOMG\val.txt
baseline no_ncc @ threshold 0.001:
  TP=1834 FP=166 FN=166 precision=0.9170 recall=0.9170 F1=0.9170
samurai_gate @ threshold 0.001 under the same FP budget:
  TP=1848 FP=50 FN=152 precision=0.9737 recall=0.9240 F1=0.9482
delta:
  recall +0.0070, precision +0.0567, FP -116
```

Smoke result path:

```text
U:\URAP_cold_storage\Desktop_URAP\artifacts\detector_first_temporal_recovery\matched_fp_no_ncc_vs_samurai_gate_smoke2000\compare.json
```

This is encouraging but not yet the final claim. The full 20762-frame validation
must still be run before claiming the method beats the current baseline.

## Baseline Beating Strategy

The target is not just higher raw recall. The target is:

```text
higher recall than TransVisDrone / YOLOMG / E104 at matched false positives
```

Required ablation ladder:

```text
E104 v2 only
E104 v2 + low-threshold top-K
E104 v2 + constant-velocity motion rescoring
E104 v2 + SAMURAI-style selective memory gate
E104 v2 + memory-guided zoom re-detection
E104 v2 + memory gate + zoom + hard reset
```

Claim gate:

- NPS-style: AP@0.5, recall, precision at matched score threshold.
- AOT-style: FPPI/FAR/HFAR fixed first, then maximize EDR/AFDR.
- DJI/tiny heldout: matched false positives, compare recovered missed frames.

The method only counts as better if it improves recall without increasing false
alarms beyond the selected baseline level.

## Native Video Mainline Direction

The detector-first version is no longer the main research path for the paper
claim. It remains useful as a baseline, diagnostic tool, and fallback ablation,
but the requested method must be detector-free:

```text
video clip
  -> frame/motion channels
  -> dense multi-scale frame tokens
  -> motion-aware clean memory selection
  -> memory-conditioned cross-attention over the current full frame
  -> dense tiny-object heatmap + bbox/action chunk head
  -> threshold / NMS / matched-FP evaluation
```

This is closer to an Octo-style video/action architecture, still with no
language branch. The critical SAMURAI transfer is not the tracker interface; it
is the idea that clean historical memory can condition the current frame so that
re-acquisition behaves like detection. In our setting, that means the heatmap
itself must be memory-conditioned, not just the bbox regressor.

The 2026-06-23 native-video code path now has this split:

```text
current s006:
  dense stride-4 video transformer
  SAMURAI-style frame memory selection
  conv objectness head on current-frame feature map

next mainline:
  dense stride-4 video transformer
  SAMURAI-style frame memory selection
  pooled clean-memory bank
  current-frame tokens cross-attend to clean memory
  conv objectness head on memory-conditioned feature map
```

The core architectural change versus the plugin route:

- The detector is no longer an external frozen source.
- The network learns proposal/detection, motion consistency, and memory quality
  jointly.
- The action residual head becomes auxiliary supervision.
- The memory quality head decides what observations can condition future frames.
- The objectness heatmap is computed after memory conditioning, so the model can
  use cross-frame evidence to identify a tiny UAV that is ambiguous in one frame.

Concrete code mapping:

- `qstr_dronedet/native_video_detector/model.py`
  - `memory_attention="pooled_cross"` adds a compressed clean-memory bank.
  - Current dense tokens query that bank through cross-attention.
  - `dense_obj_source="conv"` now operates on the memory-conditioned map.
- `tools/train_native_video_detector.py`
  - exposes `--memory-attention pooled_cross` and `--memory-slots`.
- `tools/export_native_video_predictionsgt.py`
  - reloads the same architecture from checkpoint config for evaluation.
- `tools/start_native_video_detector_train_detached.ps1`
  - records these settings in detached-run metadata.

## Milestones

1. Detector-free native video SAMURAI-style MVP.
   - Run the `pooled_cross` memory-attention model on the full training split.
   - Evaluate only held-out validation/test splits.
   - Compare against TransVisDrone at matched false positives and mAP/recall.
   - Target: TransVisDrone-level first, then +10% relative improvement on the
     selected paper metric without exceeding its false-positive level.

2. Detector-first SAMURAI-style baseline/diagnostic.
   - Finish memory-quality gate and diagnostics.
   - Sweep memory thresholds at matched false-positive rate.
   - Compare against E104/YOLOMG/TransVisDrone predictions on the same split.

3. Better motion prior.
   - Replace simple constant-velocity score with a real Kalman covariance score.
   - Add action-chunk residual likelihood as a second motion term.
   - Keep the simple motion prior as an ablation baseline.

4. Memory-guided zoom for `dji-tiny`.
   - Trigger zoom when detector misses, target is tiny, or motion score is high
     but detector score is low.
   - Remap crop detections back to full-frame coordinates.
   - Merge and score through the same memory gate.

5. Native video detector scale-up.
   - Train full data mixture, not partial cherry-picked subsets.
   - Add memory/action tokens to the video transformer.
   - Report against TransVisDrone/YOLOMG/E104 on held-out test splits only.

## Risks

- A tracker-only route can hallucinate targets. Mitigation: memory state can
  guide candidate selection, but final output still needs current-frame evidence.
- NCC can drift. Mitigation: NCC is support evidence, not an unconditional memory
  write.
- Kalman can over-trust wrong velocity. Mitigation: hard reset and uncertainty
  inflation after misses.
- SAM2 masks may be too heavy or unstable for tiny UAVs. Mitigation: bbox/ROI
  memory first; SAM2/SAMURAI mask branch remains optional for offline research.

## Decision

Use SAMURAI as the conceptual backbone for the detector-free research direction:

```text
native video motion-aware memory for tiny-UAV detection
```

The paper-facing final system is a native video/action/memory detector. The
detector-first system is kept only as a comparison line and an engineering
diagnostic, not as the claimed method.

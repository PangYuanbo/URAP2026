# SAMURAI NPS Ablation Report

Date: 2026-06-25

## Question

This experiment separates four possible sources of the NPS tracking result:

1. SAM 2 cross-frame video memory and propagation.
2. SAMURAI motion-aware mask/memory selection.
3. NPS domain fine-tuning.
4. Mask output versus a learned bbox readout from the same frozen video state.

## Protocol

- Test split: 99 NPS tracks, 16,385 frames, 16,357 visible frames.
- Train split used only for the bbox readout: 306 tracks, 38,039 frames.
- Every test track receives the ground-truth box only on frame 0.
- All tracking rows use the same base-plus checkpoint family, image size, test split and metric implementation.
- Primary metrics: success AUC, mean IoU, success at IoU 0.5 and center precision at 20 px.
- Significance: paired bootstrap over the 99 test tracks, 10,000 resamples, fixed seed 20260625.
- No test-set threshold sweep or checkpoint selection was performed.

## Controlled Rows

| Row | Video memory | SAMURAI motion selection | NPS fine-tune | AUC | mIoU | IoU>=0.5 | P@20 |
|---|---|---|---|---:|---:|---:|---:|
| Image SAM + previous-box prompt | No | No | No | 0.0917 | 0.0501 | 0.0581 | 0.0881 |
| SAM2 video | Yes | No | No | 0.5954 | 0.5975 | 0.7810 | 0.8981 |
| SAMURAI video | Yes | Yes | No | 0.6014 | 0.6038 | 0.7907 | 0.8975 |
| **SAM2 video + NPS** | Yes | No | Yes | **0.6765** | **0.6840** | **0.8749** | **0.9427** |
| SAMURAI video + NPS | Yes | Yes | Yes | 0.6572 | 0.6635 | 0.8604 | 0.9225 |

The image-only row still uses the SAM2 image encoder. It removes cross-frame video memory and re-prompts every frame with the previous predicted box. Therefore this row measures the complete cross-frame memory/propagation contribution, not the removal of visual features.

## Contribution 1: SAM2 Video Memory

SAM2 video minus framewise image-box, both zero-shot:

- Frame-weighted AUC: +0.5037.
- Mean IoU: +0.5474.
- Success@0.5: +0.7229.
- P@20: +0.8100.
- Mean paired sequence AUC delta: +0.4578.
- 95% CI: [+0.4136, +0.5001].
- Improved/tied/worse tracks: 93 / 0 / 6.

Conclusion: the cross-frame SAM2 video memory and propagation mechanism is the dominant contribution, with a large and statistically clear gain.

## Contribution 2: SAMURAI Motion Selection

The stock and SAMURAI configurations use the same base model. The SAMURAI configuration only adds motion/Kalman mask selection and memory-bank gates.

Zero-shot SAMURAI minus stock SAM2 video:

- Frame-weighted AUC: +0.0060.
- Mean paired sequence AUC delta: +0.0079.
- 95% CI: [-0.0004, +0.0175].
- Improved/tied/worse tracks: 65 / 8 / 26.

After NPS fine-tuning, SAMURAI minus stock SAM2 video:

- Frame-weighted AUC: -0.0193.
- Mean paired sequence AUC delta: -0.0094.
- 95% CI: [-0.0204, -0.0016].
- Improved/tied/worse tracks: 30 / 4 / 65.

Conclusion: the default SAMURAI motion-aware selection is not proven beneficial zero-shot and is significantly harmful after NPS fine-tuning. The likely interpretation is that its fixed motion and memory gates are not calibrated to NPS tiny targets. A later recalibration must be selected on train/validation data, not on this test set.

## Contribution 3: NPS Domain Fine-Tuning

NPS fine-tuned stock SAM2 minus zero-shot stock SAM2:

- Frame-weighted AUC: +0.0811.
- Mean IoU: +0.0865.
- Mean paired sequence AUC delta: +0.0708.
- 95% CI: [+0.0488, +0.0935].
- Improved/tied/worse tracks: 71 / 1 / 27.

NPS fine-tuned SAMURAI minus zero-shot SAMURAI:

- Frame-weighted AUC: +0.0558.
- Mean paired sequence AUC delta: +0.0535.
- 95% CI: [+0.0342, +0.0730].
- Improved/tied/worse tracks: 65 / 1 / 33.

Conclusion: NPS domain fine-tuning provides a significant gain under both memory-selection modes. The best tested system is NPS-fine-tuned stock SAM2 video, not the default SAMURAI motion configuration.

## Contribution 4: Mask Output vs BBox Readout

SAM2/SAMURAI does not contain a native bbox tracking head. The controlled bbox ablation therefore freezes the NPS-fine-tuned SAMURAI video model and trains a 100,740-parameter MLP readout using:

- the selected 256-dimensional SAM mask-decoder object pointer;
- the normalized previous box;
- a four-value box delta target.

The readout used all 37,631 valid non-initial training rows for 80 epochs. Smooth-L1 training loss decreased from 0.6550 to 0.3468. The test split was not used for training or model selection.

| Output mode | AUC | mIoU | IoU>=0.5 | P@20 |
|---|---:|---:|---:|---:|
| **Mask-to-box reference** | **0.6588** | **0.6649** | **0.8604** | **0.9225** |
| BBox readout, mask-conditioned previous box | 0.4551 | 0.4495 | 0.4970 | 0.8986 |
| BBox readout, GT-conditioned previous box | 0.4654 | 0.4611 | 0.5134 | 0.9279 |
| BBox readout, fully autoregressive | 0.0576 | 0.0111 | 0.0082 | 0.0553 |

Primary head-only comparison: mask-to-box minus mask-conditioned bbox readout:

- Frame-weighted AUC: +0.2037.
- Mean IoU: +0.2154.
- Success@0.5: +0.3634.
- P@20: +0.0239.
- Mean paired sequence AUC delta: +0.2428.
- 95% CI: [+0.2121, +0.2737].
- Mask/bbox wins: 95 / 4 tracks.

Even with the previous ground-truth box supplied, the bbox readout remains 0.1934 AUC below the mask output, so the head-only gap is not explained only by recursive drift. Fully autoregressive bbox prediction collapses because each box error changes the next input distribution; this is a deployment-stability result, not the primary head-isolation estimate.

The mask reference differs from the original fine-tuned SAMURAI AUC (0.6572 versus 0.6588) only because this head ablation explicitly replaces each track's first mask-derived box with the common ground-truth initialization box.

## Overall Finding

The original statement that the complete SAMURAI route works well needs refinement:

- Strongly supported: SAM2 cross-frame video memory.
- Strongly supported: NPS domain fine-tuning.
- Supported within this frozen-feature readout experiment: mask output is substantially stronger and more stable than the learned bbox readout.
- Not supported: default SAMURAI motion-aware memory selection as the source of the gain.

The evidence points to **NPS-fine-tuned SAM2 video memory with mask output** as the current best architecture. SAMURAI's motion gate should be treated as an optional component requiring NPS-specific validation and recalibration.

## Scope and Limitations

- This is single-object tracking with a frame-0 target prompt, not automatic drone detection.
- The bbox comparison is against a controlled lightweight readout from the frozen SAM mask-decoder object pointer, not against every possible bbox transformer or detector.
- The video-memory ablation removes the complete cross-frame memory path; it does not separately isolate each memory-attention submodule.
- Weak NPS masks are derived from boxes, so this experiment shows an architectural/output advantage under the present supervision, not an advantage from manually annotated pixel-perfect masks.

## Reproducibility Artifacts

- Machine-readable summary: `artifacts/samurai_ablation/summary.json`
- Full bbox metrics: `U:\URAP_runs\samurai\ablation_bbox_readout_finetuned1_test\metrics.json`
- BBox predictions: `U:\URAP_runs\samurai\ablation_bbox_readout_finetuned1_test\metrics.predictions.npz`
- Merged train features: `U:\URAP_runs\samurai\ablation_feature_train_finetuned1_merged\features.npz`
- Test features: `U:\URAP_runs\samurai\ablation_feature_test_finetuned1\features.npz`
- BBox checkpoint: `U:\URAP_models\samurai\bbox_readout_finetuned1.pt`

## Run Incident

Train feature shard 2 (PID 53648) stopped without traceback or CUDA OOM after 54/102 completed sequences. It was observed stopped at approximately 14:37 PDT and explicitly resumed at 14:37:44 PDT as PID 54512. Resume used only complete per-sequence chunks; the partially processed sequence was recomputed. The resumed job completed 102/102 and the final merge verified all 306 global sequence IDs and 38,039 frames.

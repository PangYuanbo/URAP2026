# URAP Modal Training Assets

Workspace: `ybpang-1`

## Core code and weights
- `urap-code-artifacts-v1` -> mount `/code`
- `urap-model-weights-v1` -> mount `/weights`

## NPS
- `urap-nps-formatted-v1` -> TransVisDrone-formatted NPS
- `urap-nps-yolomg-v1` -> YOLOMG RGB/motion/labels
- `urap-nps-motion-original-v1` -> original motion control
- `urap-nps-motion-variants-v1` -> slow_0p5, fast_2x, accelerate_g2, decelerate_g2

Validated counts:
- train: 51,951 frames/labels/masks
- val: 5,944 frames/labels/masks
- test: 12,355 frames/labels/masks

## ARD100
- `urap-ard100-raw-v1` -> 65 train + 35 test videos and annotations.zip
- `urap-ard100-yolomg-train-v1` -> mount `/data_train`
- `urap-ard100-yolomg-eval-v1` -> mount `/data_eval`
- `urap-ard100-transvisdrone-links-v1` -> mount `/data_tvd` together with `/data_train` and `/data_eval`

Validated YOLOMG counts:
- train: 106,734 RGB + 106,734 motion + 106,734 labels
- val: 20,762 RGB + 20,762 motion + 20,762 labels
- test: 71,608 RGB + 71,608 motion + 71,608 labels

Validated TransVisDrone links:
- train: 106,734 frames/labels, 55 videos
- val: 20,762 frames/labels, 10 videos
- test: 71,608 frames/labels, 35 videos
- broken links: 0

## AOT part1
- `urap-aot-part1-raw-v1` -> mount `/aot`
- 172 flights
- 206,181 images
- image bytes: 506,519,912,094
- groundtruth SHA256: `a3b5867ce300f4b97c5e6e0d8711bc1ac745f07e6dd439bb748bf3324bd84f77`

The large local TransVisDrone AOT layout and SAMURAI folders are derived caches/hardlink views. Rebuild them from AOT raw data, code, and uploaded weights rather than duplicating hundreds of GiB.

## Readiness evidence
- `artifacts/modal_ard100_aot_ready_audit_final.txt`
- `artifacts/modal_ard100_transvisdrone_audit.txt`
- `artifacts/modal_final_training_readiness_final.txt`

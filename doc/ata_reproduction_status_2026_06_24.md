# ATA Reproduction Status - 2026-06-24

## Scope

Target paper: *ATA: A Benchmark for Vision-Language Tracking in Air-to-Air Counter-UAV of Tiny Drones* (`drones-10-00429.pdf`, published 2026-06-02).

The immediate target is the paper's BBox-only protocol: first-frame ground-truth box plus subsequent video frames, with no language input.

## Paper Protocol

- Train split: 40 sequences (`uav-m1` to `uav-m22`, `uav-s1` to `uav-s18`).
- Test split: 10 sequences (`uav-m23` to `uav-m28`, `uav-s19` to `uav-s22`).
- Training: initialize from each tracker's official pretrained weights and retrain on ATA train only.
- Epochs: 50 for every method.
- Template crop: 192 x 192.
- Search crop: 384 x 384.
- Checkpoint selection: last epoch; ATA test is not used for tuning or selection.
- Metrics: success AUC (primary), OP50, center precision, normalized precision, FPS.

Paper BBox-only baselines:

| Method | AUC | OP50 | Precision | Normalized precision | FPS |
|---|---:|---:|---:|---:|---:|
| OSTrack | 36.11 | 37.84 | 65.55 | 24.80 | 61.32 |
| SeqTrack | 44.55 | 48.43 | 79.74 | 36.18 | 50.00 |
| AQATrack | 36.10 | 36.31 | 70.36 | 25.53 | 55.67 |
| MCITrack | 41.09 | 42.69 | 76.97 | 30.69 | 32.95 |

## Public Release Audit

Official repository: `https://github.com/kkbushi/ATA`, commit `cb42ce14f91241bcd47f835641ba5db21cd42a51`.

Verified contents:

- All 50 `groundtruth.txt` files are present.
- All 50 `language.txt` files are present.
- No image or video file exists in the current branch, any remote branch, or any repository history commit.
- No official ATA training code, AFTE implementation, model configuration, checkpoint, or evaluation script is published.
- The repository README still says the complete dataset will be published after organization and article publication.

Annotation audit:

- Train annotations: 30,753 frames.
- Test annotations: 7,344 frames.
- Total annotations: 38,097 frames.
- Paper-reported total: 38,094 frames. The public annotations contain three more rows than the paper reports.
- Mean width: 31.1748 px, matching the paper's 31.175 px.
- Mean height: 16.8449 px, matching the paper's 16.845 px.
- Mean area: 623.8500 px^2, matching the paper's 623.850 px^2.
- Six boxes in `test/uav-s22`, frames 935-940, extend above the image and have negative y coordinates.

Authoritative audit output: `artifacts/ata_reproduction/release_audit.json`.

## Implemented Reproduction Infrastructure

- `qstr_dronedet/ata_benchmark.py`
  - Validates the official split and release completeness.
  - Computes macro sequence-average AUC, OP50, P@20, normalized precision AUC, and mean IoU.
  - Materializes ATA into the existing SAMURAI/LaSOT-style layout once images are available.
- `tools/audit_ata_release.py`
- `tools/eval_ata_predictions.py`
- `tools/prepare_ata_samurai_dataset.py`
- `tests/test_ata_benchmark.py`

Verification:

- Four metric/parser unit tests pass with `python -m unittest tests.test_ata_benchmark -v`.
- A complete ten-sequence oracle test using ATA test ground truth as predictions returns 1.0 for every metric over 7,344 frames.
- The materializer correctly refuses to produce a runnable dataset when images are missing.

## SeqTrack Reproduction Chain

Official Microsoft VideoX/SeqTrack code was checked out at commit
`1744b2159da3cbba370f4e4b8285d4e13bdba157`. The official SeqTrack-B384
checkpoint was downloaded and verified:

- Path: `U:/URAP_models/seqtrack/train/seqtrack/seqtrack_b384/SEQTRACK_ep0500.pth.tar`
- Size: 364,742,001 bytes
- SHA-256: `003802C5AECC1FC69C7407382B8254405C64898A9073FECF913D21BFA953C29D`
- Official B384 smoke test: strict checkpoint load and one CUDA tracking step succeeded on RTX 5090.
- Measured public model size: 91.169M parameters, while the ATA paper reports 87.15M for SeqTrack.

The paper changes the template from the public B384 model's 384 pixels to
192 pixels. That changes the encoder position embedding from `[1, 1152, 768]`
to `[1, 720, 768]`. Since the paper does not publish its conversion code, the
implemented auditable adaptation preserves the 24 x 24 search grid and
bicubic-interpolates only the template grid from 24 x 24 to 12 x 12.

- Adapted checkpoint: `U:/URAP_models/seqtrack/ata_init/seqtrack_b384_template192_init.pth.tar`
- Size: 363,419,561 bytes
- SHA-256: `EB67B56410CBA197A88D588100B65AF30E61F5991467EF44D0A4214DA5053592`
- Strict load verified: yes
- CUDA forward with 192/384 crops verified: yes
- Adapted public architecture size: 90.838M parameters
- Peak allocated CUDA memory in the one-step smoke test: 0.437 GiB

This parameter-count discrepancy is additional evidence that the exact
SeqTrack configuration used for Table 3 is not present in the public release.
The adapted model is a documented best-effort reconstruction, not a claim of
bit-exact reproduction.

Implemented SeqTrack support:

- `third_party/VideoX/SeqTrack/lib/train/dataset/ata.py`: fixed 40-sequence ATA train adapter.
- `third_party/VideoX/SeqTrack/experiments/seqtrack/seqtrack_b384_ata.yaml`: 50 epochs and 192/384 crops.
- `tools/adapt_seqtrack_checkpoint_for_ata.py`: deterministic positional-embedding conversion.
- `tools/run_seqtrack_ata.py`: fixed-split inference and prediction export.
- `tools/start_seqtrack_ata_eval_detached.ps1` and `tools/monitor_seqtrack_ata_eval.ps1`.
- `tools/start_seqtrack_ata_train_detached.ps1` and `tools/monitor_seqtrack_ata_train.ps1`.

The training configuration retains the official B384 optimizer, batch size,
two-template setup, and 60,000 samples per epoch. Those values are reasonable
minimal-change assumptions; the ATA paper does not disclose its
samples-per-epoch or learning-rate schedule. The start script refuses to
create a PID when any image sequence is incomplete. This no-launch guard was
verified against the current annotation-only release.

## Existing Methods: What Is and Is Not Comparable

The strongest completed local first-frame-prompt tracking result is SAMURAI
base-plus on the NPS test split:

| Method | Dataset | Training | Success AUC | IoU >= 0.5 | P@20 |
|---|---|---|---:|---:|---:|
| SAMURAI base-plus | NPS test | zero-shot | 60.14 | 79.07 | 89.75 |
| SAMURAI base-plus | NPS test | one NPS epoch | 65.72 | 86.04 | 92.25 |

These numbers cannot be compared numerically with ATA's SeqTrack AUC 44.55:
the datasets, sequence lengths, target ambiguity, and train/test distributions
are different. They establish that the SAMURAI path is runnable and is the
first prior method to evaluate once ATA images are available; they do not show
that it beats the paper.

The previous YOLOMG/TransVisDrone detector, detector-first temporal reranker,
and native video/action-chunk model report detection precision/recall or mAP.
ATA Table 3 is single-object tracking initialized by the first-frame ground
truth box. Reporting those detector scores beside ATA AUC would be invalid.
For a fair ATA comparison they must emit one box per frame after receiving the
same first-frame box, then use the ATA evaluator.

Planned ATA rows and present state:

| Method | ATA runner state | Actual ATA score |
|---|---|---|
| Official SeqTrack-B384 zero-shot | Ready; CUDA smoke verified | Blocked by missing images |
| Reconstructed SeqTrack 192/384, 50 epochs | Dataset/config/init/start/monitor ready | Blocked by missing images |
| SAMURAI base-plus zero-shot | Converter and evaluator ready | Blocked by missing images |
| NPS-fine-tuned SAMURAI transfer | Converter and evaluator ready | Blocked by missing images |
| Native video/action model | Requires first-frame tracking output adapter | Not yet comparable |
| Detector-first motion-memory path | Requires one-box-per-frame tracking mode | Not yet comparable |

## Planned Fair Comparison

Once the images are available, run in this order:

1. SeqTrack official pretrained B384 zero-shot on ATA test.
2. SAMURAI/SAM2.1 base-plus zero-shot on ATA test.
3. Existing NPS-fine-tuned SAMURAI checkpoint on ATA test, marked explicitly as cross-dataset transfer.
4. SeqTrack initialized from official weights and fully retrained for 50 ATA epochs using 192/384 crops.
5. SAMURAI fully fine-tuned on all 40 ATA train sequences and evaluated only on the 10 test sequences.
6. Existing native video/action model trained on all ATA train sequences, using the same fixed test split.

All trackers must receive the same first-frame ground-truth box. Results will be reported with the paper metrics and the local supplementary metrics, without comparing detection mAP/recall to tracking AUC.

## Current Status

Reproduction is **NOT RUNNING**: `0/10` ATA test sequences and `0/50` training
epochs have run. There is no ATA evaluation PID and no ATA training PID. The
public release contains 0/38,097 image frames, and the local dataset drives and
download folders were also checked without finding another ATA archive. No
training process was started or silently substituted with NPS data.

The next external requirement is the 38,097 ATA image frames (or the original 50 videos with an exact frame extraction protocol). After those files are obtained, the prepared audit and conversion tools can validate the release before any GPU job starts.

# SAMURAI TVD-format ARD100 Ablation Report

Date: 2026-06-26
Status: blocked by an external USB storage disconnect. Zero-shot rows are complete; ARD100 fine-tuning and downstream rows are NOT RUNNING.

## Dataset protocol

- Domain source: TVD-format ARD100, not NPS.
- Official video split: 55 train / 10 validation / 35 test videos.
- Train: 108,377 timeline frames, 106,891 visible target frames, 1,486 empty frames.
- Validation: 21,459 timeline frames, 20,767 visible target frames, 692 empty frames.
- Test: 72,631 timeline frames, 71,633 visible target frames, 998 empty frames.
- Train, validation and test video IDs are disjoint.
- Full 1920x1080 backgrounds are retained.
- Weak training masks are rectangles generated from continuity-selected ARD100 annotations.
- Every test video receives the ground-truth box only on frame zero.

## Full-data training protocol

The stock SAM2 training loader samples only one random 8-frame clip per video per epoch. The ARD100 configuration instead uses a window-indexed dataset view:

- 13,406 windows per full data epoch.
- 8 original-timeline frames per window.
- Every window starts on a visible target frame.
- Empty frames inside a window remain in the original temporal position.
- All 106,891 visible train frames are covered at least once per full data epoch.
- Images and masks are not duplicated for the window index.

After the storage failure, one full data epoch is divided into four deterministic phases. Each phase consumes a disjoint quarter of the same fixed permutation, and all four phases together still cover 13,406/13,406 windows exactly once. A recoverable checkpoint is written after each phase to internal C: storage.

## Controlled ablations

1. Framewise image SAM with the previous predicted bbox as prompt.
2. Stock SAM2 video memory and propagation.
3. SAMURAI motion-aware selection and memory gates.
4. ARD100 full-frame domain fine-tuning, evaluated with stock and SAMURAI configs.
5. Frozen-video-state learned bbox readout versus mask-to-box output.

## Completed zero-shot results

All values below are canonical merged results over the complete 35-video ARD100 test split.

| Row | Success AUC | Mean IoU | IoU >= 0.5 | Precision @ 20 px |
|---|---:|---:|---:|---:|
| Framewise image-box | 0.059390 | 0.013352 | 0.014518 | 0.022169 |
| Stock SAM2 video memory | 0.144518 | 0.105710 | 0.132285 | 0.152583 |
| SAMURAI motion-aware | 0.150840 | 0.112507 | 0.137618 | 0.160694 |

Observed deltas before statistical bootstrap:

- Video memory versus framewise image-box: +0.085128 Success AUC.
- SAMURAI motion-aware selection versus stock SAM2: +0.006321 Success AUC.

These values show a large benefit from temporal video memory and a smaller positive average contribution from SAMURAI motion-aware selection. They do not yet establish per-sequence significance; paired bootstrap remains part of the final summary stage.

## Fine-tuning interruption

The formal full-data run started at 2026-06-25 20:46:59 PDT with launcher PID 59576 and compute PID 58472. Its last completed logged unit was 10,280/13,406 windows at 2026-06-26 00:17:54 PDT, with cumulative average training loss 3.27. No checkpoint had yet been written under the original one-checkpoint-per-epoch policy.

At 2026-06-26 00:27:52 PDT, Windows recorded disk Event ID 51 and NTFS Event ID 140 for the USB-attached Fanxiang S880 2TB volume U:. Training stderr concurrently reported WinError 433 (device does not exist). The drive then disappeared from Get-Disk and present PnP enumeration. Both training processes exited, and GPU memory fell from about 31 GB to about 1.8 GB.

Current authoritative state:

- Fine-tuning: NOT RUNNING, 10,280/13,406 completed, no usable ARD100 fine-tuned checkpoint.
- Fine-tuned stock/SAMURAI test evaluation: NOT RUNNING, 0/35 new sequences.
- Full train/test feature export: NOT RUNNING.
- Learned bbox readout: NOT RUNNING.
- Cause: external USB storage disconnect, not model loss divergence, CUDA OOM, or architecture failure.

## Storage recovery status

At 2026-06-26 22:18 PDT the Fanxiang S880 2TB device re-enumerated as Disk 2 and U:. Representative ARD100 train/validation/test frames and masks and the base checkpoint were readable. However, Windows reports HealthStatus=Warning, OperationalStatus=Full Repair Needed, the NTFS dirty bit is set, and Event ID 98 requires an offline CHKDSK /F or Repair-Volume operation.

The verified base SAM2 checkpoint was copied to internal C: storage and both copies matched SHA-256 A2345AEDE8715AB1D5D31B4A509FB160C5A4AF1970F199D9054CCFB746C004C5. Restart configs now load that protected C: copy.

An unrelated agent is actively uploading ARD100 data from U: to Modal. Its processes are not terminated or modified. Offline repair and restart training remain gated until all U: readers exit naturally. A detached read-only watcher records U: users at `artifacts/samurai_runs/wait_for_u_volume_idle.progress.json`.

## Recovery changes

- Dataset loading now fails immediately when WinError 433 occurs or a configured data root disappears, instead of retrying random samples up to 100 times.
- The ARD100 epoch is split into four deterministic phases, retaining full-data coverage.
- Phase checkpoints are written to C:/Users/aaron/Desktop/URAP/artifacts/samurai_checkpoints/finetune_base_plus_ard100_fullframe_stage1_restart1.
- The training coordinator requires all four phase checkpoints and process exit before reporting completion.
- The post-finetune pipeline reads the protected C: checkpoint. All restart training, evaluation, feature, bbox, summary, PID, and log names are isolated with a `restart1` suffix so they cannot resume or merge old partial outputs.

Before restart, U: must enumerate as Fanxiang S880 2TB, the ARD100 dataset/model roots must exist, representative files must be readable, and the existing run directory must be audited. Restart must be explicit with a new PID and log record; no silent restart is allowed.

## Reproducibility artifacts

- Dataset root: `U:\URAP_datasets\TransVisDrone\ARD100\SAMURAI`
- Failed-run training config: `third_party/samurai/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_ARD100_fullframe_stage1.yaml`
- Restart training config: `third_party/samurai/sam2/sam2/configs/sam2.1_training/sam2.1_hiera_b+_ARD100_fullframe_stage1_restart1.yaml`
- Pipeline launcher: `tools/start_samurai_ard100_restart1_pipeline_detached.ps1`
- Pipeline monitor: `tools/monitor_samurai_ard100_restart1_pipeline.ps1`
- Restart preflight: `tools/check_samurai_ard100_restart_ready.ps1`
- Fine-tune child launcher: `tools/start_samurai_finetune_detached.ps1`
- Training coordinator: `tools/sequence_samurai_ard100_finetune.ps1`
- Post-finetune coordinator: `tools/sequence_samurai_ard100_post_finetune.ps1`
- Storage failure audit: `artifacts/samurai_runs/ard100_fullframe_stage1_storage_disconnect_20260626.json`
- Zero-shot shard monitor: `tools/monitor_samurai_ard100_zero_shot_shards.ps1`
- Shard merger: `tools/merge_samurai_eval_shards.py`
- Final summary: `tools/summarize_samurai_ard100_ablation.py`

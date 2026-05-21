# URAP2026

URAP2026 is the lightweight research workspace for UAV small-object detection experiments.

The current active prototype is **QSTR-DroneDet**: a quality-aware, two-stage tiny drone detection and recognition scaffold. It separates candidate generation from recognition so experiments can diagnose whether a miss came from proposal failure, recognition failure, motion alignment failure, feature loss, or weak single-frame evidence.

This repository intentionally tracks:

- QSTR-DroneDet source code in `qstr_dronedet/`
- synthetic/unit tests in `tests/`
- empty data templates in `data_templates/`
- experiment and reproduction notes in `doc/`
- project reports in `reports/`
- Windows/PowerShell and Python runner scripts in `tools/`
- repository operating rules in `AGENTS.md`

It intentionally does not track:

- raw datasets, extracted videos, frames, or annotations
- model weights/checkpoints
- generated artifacts and training outputs
- third-party cloned paper repositories under `papers/` and related local folders
- Python virtual environments

## Repository Layout

```text
qstr_dronedet/      QSTR-DroneDet Python package
tests/              Synthetic tests and CLI smoke tests
data_templates/     CSV templates for real video recording and annotation
doc/                Reproduction notes and experimental logs
reports/            Professor-facing summaries and paper/dataset notes
tools/              PowerShell/Python runners for repeatable experiments
```

## Key Local Dependencies

The full local workspace uses external code/data checked out or downloaded beside this repo:

- ESOD official code: `papers/ESOD`
- TransVisDrone code: `papers/TransVisDrone`
- NPS/Dogfight annotations/code: `datasets/Drone-Detection`
- Li-TETC / Fast-and-Robust UAV-to-UAV baseline: `Fast-and-Robust-UAV-to-UAV-Detection-and-Tracking`
- ARD100 data: `D:\URAP_datasets\ARD100`
- AOT data: `D:\URAP_datasets\AOT`

These paths are local workspace conventions, not git-tracked assets.

## Current Focus

The current QSTR experiment line combines:

1. Stage A class-agnostic tiny-object proposal generation with YOLO-P2, motion proposals, tracker proposals, and optional fallback detector proposals.
2. Stage B object recognition using image crops, feature ROI crops, temporal tubes, motion/alignment metadata, and tracker metadata.
3. Motion quality logging through `q_H` / alignment quality so bad homography or fast ego-motion does not blindly dominate fusion.
4. Separate stable and hard-recovery inference profiles:
   - stable/default: lower false positives, old YOLO-P2 primary detector, no verified objectness boost.
   - hard-recovery: optional fallback detector plus stricter gate for tiny / low-objectness recovery cases.
5. Frozen held-out Anti-UAV sequence checks before deciding whether to scale training.

Useful entry points:

- `python -m qstr_dronedet.cli --help`
- `tools/run_qstr_stable_profile.ps1`
- `tools/run_qstr_hard_recovery_profile.ps1`
- `tools/run_qstr_frozen10_profile_benchmark.ps1`
- `tools/run_qstr_tests.ps1`
- `doc/qstr_sanity_static_fast_tests_2026_05_20.md`
- `doc/qstr_real_video_data_protocol.md`
- `doc/official_datasets_and_metrics.md`
- `doc/progress_report_for_professor.md`

Older ESOD / TransVisDrone / YOLOMG reproduction notes remain in `doc/` and `tools/` for baseline comparison.

## Quick Checks

From the repository root:

```powershell
python -m qstr_dronedet.cli --help
pytest tests -q
```

Run one QSTR profile on a video:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_stable_profile.ps1 `
  -Video D:\datasets\Anti-UAV300\qstr_heldout_test_visible_10seq\raw_videos\test\visible\20190925_111757_1_5\visible.mp4 `
  -Out runs\profiles\stable_example `
  -Device 0
```

Run the frozen held-out benchmark wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_qstr_frozen10_profile_benchmark.ps1 `
  -OutRoot runs\profiles\frozen10_profile_eval `
  -Device 0
```

These commands expect local datasets and model weights to exist on the machine. The repo stores scripts and code only, not the datasets or checkpoints.

## Detached Long Runs

Long-running jobs must be launched through the `tools/start_*_detached.ps1` scripts and monitored by matching `tools/monitor_*` scripts. Status reports should be based on PID, output timestamps, progress counters, and GPU signal when relevant.

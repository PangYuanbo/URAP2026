# URAP2026

URAP2026 is the lightweight orchestration and documentation layer for the local UAV small-object detection research workspace.

This repository intentionally tracks:

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

The current experiment line combines:

1. NPS/Dogfight-style motion-boundary proposal generation.
2. ESOD-style ROI cropping and high-resolution enlargement.
3. ESOD fine-tuning on generated NPS motion ROI data.
4. Cross-dataset robustness evaluation across NPS and ARD100.

Useful entry points:

- `tools/build_motion_esod_rois.py`
- `tools/run_esod_nps_motion_pipeline.py`
- `tools/start_esod_nps_motion_pipeline_detached.ps1`
- `tools/monitor_esod_nps_motion_pipeline.ps1`
- `doc/repro_esod.md`
- `doc/official_datasets_and_metrics.md`
- `doc/progress_report_for_professor.md`

## Detached Long Runs

Long-running jobs must be launched through the `tools/start_*_detached.ps1` scripts and monitored by matching `tools/monitor_*` scripts. Status reports should be based on PID, output timestamps, progress counters, and GPU signal when relevant.

# Saved AOT / NPS / ARD motion comparisons

This pipeline runs preprocessing only: no training, detector inference, or tracking evaluation.

## Coverage

- NPS: all 50 locally available clips. Original MOVs for 36 training clips; contiguous cached images for 4 validation and 10 test clips. Image-sequence playback uses a nominal 29.97 FPS, not recovered timestamps.
- ARD: all 100 original MP4 videos, assigned to the existing 55/10/35 train/validation/test split. The raw sources are staged from mounted Drive to the output disk. Cached ARD images have internal gaps and are not used as consecutive video frames.
- AOT: only the 10 locally cached flights, 11,987 frames. This is **not the complete public AOT dataset**. Frame counts and FPS are checked against `groundtruth_part0.json`.

The inventory pins the latest parallax-robust implementation, the author implementation, parameters, and pipeline source hashes. Editing pipeline source during a run causes later workers to fail the integrity check; do not edit a live production pipeline.

## Methods and controls

1. **Author FD5 reference:** Gaussian 11x11, grayscale, original YOLOMG `motion_compensate`, offsets t-2 and t+2, original uint8 addition before division, JPEG conversion. The author's unsigned wraparound is intentionally preserved rather than silently corrected.
2. **Global-only control:** the new reliable bidirectional tracker and global homography at t-1 and t+1, without local correction or adaptive suppression.
3. **Parallax raw:** the new local correction at the same temporal offsets, before suppression. This intermediate is measured, but does not have its own video.
4. **New motion:** local parallax correction plus the unchanged adaptive suppression defaults.

All methods share aspect-preserving letterbox preprocessing onto a 1920x1080 canvas, fixed random seeds for matched controls, and the same source frames. The four-panel video shows RGB, author FD5, global-only, and new motion. All motion panels have the same fixed linear x3 display gain; standalone motion videos have no display gain. Source video frames are not dropped; first/last temporal neighbors are clamped within their own sequence. Future-frame use means this is offline, not a causal tracker.

AOT-trained TVD uses temporal RGB, not a native optical-flow input. Its FD5 output is explicitly a **shared YOLOMG reference**, not an original AOT/TVD method. These standardized recomputations are also not byte-identical historical YOLOMG cache artifacts.

## Saved files

Each `<output>/<dataset>/<split>/<sequence>/` contains:

- `new_motion.mp4`: full-resolution unscaled new residual.
- `original_fd5.mp4`: full-resolution unscaled author-reference residual.
- `comparison.mp4`: synchronized four-panel visualization.
- `preview.jpg`, `report.json`, `progress.json`, and encoder logs.

The output root contains `index.html`, `summary.json`, and `manifest.json`. The catalog and dataset summaries update after each completed sequence. Raw ARD copies remain in `source_cache/ARD/`. Partial MP4s are renamed only after encoding and ffprobe frame-count/dimension verification. Completed sequences are skipped on explicit resume; interrupted sequences restart from their beginning. The output files are lossy viewing artifacts, not model-input caches.

## Interpretation

Metrics sample every 30th source frame, exclude the first/last two frames, and use the intersection of method-valid regions and nonpadding content. Author-reference failures are visibly marked and exclude that entire sample from shared metrics. Dataset summaries are pixel-weighted over sampled frames, not equal-weighted per clip. Reports retain per-frame alignment diagnostics and failure counts.

Mean intensity and active fractions above 8/255 or 32/255 measure **image-wide residual activity**, not background-only error, object preservation, optical-flow endpoint error, AP, or tracking quality. Darker output is not sufficient evidence of improvement. Compare global-only versus parallax-raw to isolate local alignment, then raw versus final to see suppression. Old versus new also differs in temporal spacing, smoothing, and author arithmetic. Ground-truth target-region evaluation remains separate work.

## Commands

Use the available Python interpreter; FFmpeg/ffprobe paths are recorded in the manifest.

```powershell
& 'C:/Users/aaron/AppData/Local/Programs/Python/Python311/python.exe' optical_flow_advanced/dataset_videos.py inventory --run-dir artifacts/dataset_motion_videos_20260903 --output-root H:/URAP_OpticalFlow_20260903 --workers 2 --opencv-threads 4
& tools/start_dataset_motion_videos_detached.ps1 -Manifest artifacts/dataset_motion_videos_20260903/manifest.json -Mode smoke -SmokeFrames 90
& tools/monitor_dataset_motion_videos.ps1 -Manifest artifacts/dataset_motion_videos_20260903/manifest.json -Mode smoke
& tools/start_dataset_motion_videos_detached.ps1 -Manifest artifacts/dataset_motion_videos_20260903/manifest.json
& tools/monitor_dataset_motion_videos.ps1 -Manifest artifacts/dataset_motion_videos_20260903/manifest.json
& tools/stop_dataset_motion_videos.ps1 -Manifest artifacts/dataset_motion_videos_20260903/manifest.json
```

After a verified stop, explicitly remove only that run's `STOP_REQUESTED` file if present, then launch with `-Resume`. Never start a second coordinator for a live manifest. Launch metadata, PID files, stdout, stderr, and progress remain in the repository's run directory. Monitoring checks OS command lines and actual counter movement, and reports GPU utilization/memory and encoder PIDs. The optical-flow calculation itself is CPU OpenCV; the RTX 5090 accelerates H.264 encoding, not LK tracking.

Focused tests (run from `optical_flow_advanced` to avoid unrelated broken root dataset links):

```powershell
& 'C:/Users/aaron/AppData/Local/Programs/Python/Python311/python.exe' -m pytest tests -q --rootdir=. --confcutdir=. -o testpaths=tests
```

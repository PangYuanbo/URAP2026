# AOT / NPS / ARD saved motion-video comparison

## Where to find the outputs

- Production video catalog: `H:/URAP_OpticalFlow_20260903/index.html`
- Automatically updated comparison: `H:/URAP_OpticalFlow_20260903/comparison_report.md`
- Machine-readable aggregate: `H:/URAP_OpticalFlow_20260903/summary.json`
- Completed real-data validation clips: `H:/URAP_OpticalFlow_20260903_validation/_smoke/`
- Active production manifest: `artifacts/dataset_motion_videos_20260903_v2/manifest.json`

Each sequence produces `new_motion.mp4`, `original_fd5.mp4`, and a synchronized four-panel `comparison.mp4`, plus a preview, metrics, progress, and encoding logs. The full inventory schedules **160 sequences / 480 output videos**. This is the scheduled total, not a claim that they have all finished.

| Dataset | Scheduled sequences | Source coverage |
|---|---:|---|
| NPS | 50 | All local clips; raw MOVs for training, contiguous cached images for validation/test |
| ARD | 100 | All raw videos, existing 55/10/35 train/validation/test split |
| AOT | 10 | 11,987 locally cached frames only; **partial public-dataset coverage** |

Cached ARD images contain temporal gaps, so consecutive raw-video frames are used instead. AOT cached frame counts match their flight metadata. Image-sequence NPS playback uses nominal 29.97 FPS; original video FPS is read separately per clip.

## What is being compared

**Original reference:** the author's YOLOMG grid-KLT/global-homography motion compensation, Gaussian 11x11 preprocessing, t-2/t/t+2 FD5 differences, original unsigned-integer addition, and JPEG conversion.

**New method:** bidirectionally checked tracking, RANSAC global alignment, robust spatially varying residual correction, symmetric t-1/t/t+1 difference, and adaptive residual suppression. The latest implementation and default parameters are pinned without retraining or changing the optical-flow algorithm.

**Matched control:** the new tracker at t-1/t/t+1 with global alignment only, without local correction or suppression. Raw parallax residuals before suppression are also measured. This separates the local-alignment effect from the suppression effect.

All methods share an aspect-preserving 1920x1080 canvas. Motion panels have identical fixed x3 visualization gain. Standalone motion videos have no gain. The original implementation's arithmetic behavior is deliberately preserved, including uint8 wraparound. This is a standardized recomputation, not reuse of historical cached masks.

**AOT clarification:** the AOT-trained TVD model uses RGB rather than its own optical-flow input. The AOT FD5 result is a shared YOLOMG reference, not a native AOT/TVD flow baseline.

## Preliminary sample measurements

These are **smoke-test observations, not dataset-level results**: one 90-frame excerpt per dataset, with statistics at source frames 30 and 60 only. Each method uses identical valid pixels within the same excerpt. Values are measured before lossy video encoding.

| Excerpt | Author FD5 mean | Global-only mean | Parallax raw mean | New final mean | Author active >8 | New active >8 |
|---|---:|---:|---:|---:|---:|---:|
| NPS Clip 023 | 3.8204 | 2.7158 | 2.3682 | 0.2550 | 8.3809% | 0.9354% |
| AOT flight 0a98e01c... | 0.8603 | 0.9048 | 0.8298 | 0.0231 | 0.6065% | 0.0518% |
| ARD Clip 145 | 0.3932 | 1.3466 | 1.3792 | 0.1925 | 0.1233% | 0.6796% |

Mean intensity uses a 0-255 scale. Active fractions count pixels above 8/255. In these samples:

- Local correction lowers mean residual relative to the matched global-only control by about **12.8% on NPS** and **8.3% on AOT**, but **increases it by 2.4% on ARD**.
- The large reduction from raw parallax residual to final output primarily reflects adaptive suppression. It must not be attributed entirely to better optical flow.
- ARD's final output has a lower overall mean than author FD5, but a **higher fraction of pixels above 8/255**. Mean intensity alone therefore does not establish superiority.
- Old-versus-new differences also reflect smoothing, temporal spacing, and arithmetic. There is no ground-truth optical-flow endpoint-error evaluation here.
- These image-wide statistics cannot distinguish useful target responses from unwanted background responses. No claim about improved cross-dataset AP, recall, tracking, or target preservation follows from this table.

Full evidence is in `H:/URAP_OpticalFlow_20260903_validation/_smoke/smoke_summary.json`. All nine generated validation videos passed frame-count/dimension checks and decoded without errors; no author-reference frame failed in these excerpts.

## Execution and recovery record

The first production batch, coordinator PID **82784**, was stopped and verified absent at **September 3, 2026, 03:16:31 PDT**, after a Windows concurrent-access error on a progress JSON file. At that point **0/160 sequences had finalized**; 491 partially processed frames across five attempted sequences were not counted as completed videos. Validation outputs were unaffected.

The fix adds bounded retries for atomic replacement and transient concurrent JSON reads. It passes **12 focused tests** and a stress check of **300 updates with three concurrent readers**, without errors. Algorithms and parameters remain unchanged. The old manifest, logs, and stopped-sequence reports are preserved rather than silently rewriting their provenance.

The explicit replacement/resume was launched on **September 3, 2026, 03:19:31 PDT**, coordinator PID **52500**, using the `v2` manifest and the same output directory. This is launch history, not a permanent assertion that the PID is still running. Check current process existence and advancing output counters with the monitor command below.

- Run metadata: `artifacts/dataset_motion_videos_20260903_v2/run.json`
- Standard output: `artifacts/dataset_motion_videos_20260903_v2/run_20260903_031931_598.stdout.log`
- Standard error: `artifacts/dataset_motion_videos_20260903_v2/run_20260903_031931_598.stderr.log`
- Recovery provenance: `artifacts/dataset_motion_videos_20260903_v2/recovery.json`

Four sequence workers use two OpenCV threads each. The local capacity probe successfully ran twelve simultaneous H.264 NVENC encoders. GPU acceleration is for video encoding; the current optical-flow calculation is CPU OpenCV, so high CUDA utilization is not expected.

```powershell
& tools/monitor_dataset_motion_videos.ps1 -Manifest artifacts/dataset_motion_videos_20260903_v2/manifest.json
& tools/stop_dataset_motion_videos.ps1 -Manifest artifacts/dataset_motion_videos_20260903_v2/manifest.json
```

Completed sequences are retained on explicit resume; partial sequences are reprocessed from their beginning. The automated catalog and comparison report cover only finalized production sequences and update as each one completes. The complete protocol and usage are in `optical_flow_advanced/DATASET_VIDEO_COMPARISON.md`.

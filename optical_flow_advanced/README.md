# URAP Advanced Optical Flow and Motion Difference

This package consolidates the complete motion-compensation progression used in URAP for moving-camera drone video. It is designed for full-length videos rather than short demonstrations.

## What is included

| Level | Method | Main purpose | Entry point |
|---|---|---|---|
| 1 | Global homography + three-frame difference | Remove dominant camera motion with a single planar transform | `baseline-homography` |
| 2 | Dual TV-L1 motion boundary + YOLOMG tail | Replace sparse tracking with dense classical flow | `nps-tvl1` |
| 3 | SEA-RAFT + compensated difference | Use learned dense flow for large displacement and weak texture | `sea-raft` |
| 4 | Double-stage motion difference | Re-align first-stage residual maps to suppress remaining structured motion | `double-stage` |
| 5 | Parallax-robust local residual field | Handle non-planar scenes, near grass, depth parallax, and display instability | `parallax-robust` |

The algorithms remain in `tools/` so existing detached jobs and experiments keep working. `run.py` provides one discoverable interface without duplicating implementations.

## Quick start

Create a Python environment and install the classical dependencies:

```bash
python -m pip install -r optical_flow_advanced/requirements-classical.txt
python optical_flow_advanced/run.py list
```

Run the recommended parallax-robust pipeline:

```bash
python optical_flow_advanced/run.py run parallax-robust -- \
  --input input.mp4 \
  --output output_parallax_robust.mp4 \
  --comparison-output output_old_vs_new.mp4 \
  --ffmpeg ffmpeg \
  --encoder libx264
```

On an NVIDIA system with an FFmpeg NVENC build, replace `libx264` with `h264_nvenc`.

Run NPS-style Dual TV-L1:

```bash
python optical_flow_advanced/run.py run nps-tvl1 -- \
  --input input.mp4 \
  --output-dir outputs/tvl1 \
  --duration-seconds 0
```

Run SEA-RAFT after installing PyTorch and cloning the upstream SEA-RAFT repository:

```bash
python -m pip install -r optical_flow_advanced/requirements-deep.txt
python optical_flow_advanced/run.py run sea-raft -- \
  --input input.mp4 \
  --output-dir outputs/sea_raft \
  --sea-raft-root /path/to/SEA-RAFT \
  --cfg /path/to/SEA-RAFT/config/eval/spring-M.json \
  --duration-seconds 0
```

`--duration-seconds 0` means the complete video for pipelines that support full-length processing.

## Recommended method

Use `parallax-robust` for the current URAP drone footage. Its chain is:

1. Bidirectional pyramidal Lucas-Kanade tracks.
2. RANSAC global homography.
3. Global warp and valid-region mask.
4. Robust local residual displacement field from homography inliers.
5. Smooth local remap for depth-dependent parallax.
6. Symmetric previous/reference/following frame difference.
7. Local texture-noise calibration and residual-floor suppression.
8. Fixed-floor gamma color mapping for stable visualization.

This is not simply “make small residuals darker.” The local field changes the geometric compensation before differencing; noise calibration and stable color scaling are later visualization and suppression stages.

## Outputs

- Full-length H.264 MP4 motion-difference video.
- Optional baseline-versus-advanced comparison MP4.
- Optional progress JSON for detached monitoring.
- Optional diagnostic frames and alignment statistics.
- Quantitative grass-noise suppression and target-preservation report through `evaluate-parallax`.

## Reproducibility

The default parallax-robust parameters are recorded in `configs/parallax_robust_default.json`. Keep these values fixed when comparing methods. Change one component at a time for ablations.

For method provenance, equations, limitations, and the distinction between published ideas and URAP engineering additions, see [METHODS.md](METHODS.md).

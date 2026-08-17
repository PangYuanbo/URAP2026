# Method progression and provenance

## 1. Global camera-motion compensation

The baseline tracks sparse points with pyramidal Lucas-Kanade optical flow, rejects inconsistent tracks with forward-backward checking, and estimates a homography with RANSAC. The previous and following frames are warped into the reference frame before a symmetric difference is computed.

Borrowed foundations:

- Lucas-Kanade optical flow: Bruce D. Lucas and Takeo Kanade, 1981.
- RANSAC: Martin A. Fischler and Robert C. Bolles, 1981.
- Projective homography and image warping are standard multiple-view geometry tools.

Limitation: one homography assumes a mostly planar scene or pure camera rotation. Drone footage with grass, buildings at different depths, and strong translation violates this assumption.

## 2. Dual TV-L1 dense flow

`nps-tvl1` uses Dual TV-L1 dense flow to estimate a motion field at every pixel. A motion-boundary map is derived from spatial flow gradients, while robust flow correspondences are still reduced to a homography for the YOLOMG-style compensation and difference tail.

Borrowed foundation:

- C. Zach, T. Pock, and H. Bischof, “A Duality Based Approach for Realtime TV-L1 Optical Flow,” 2007.

Strength: edge-preserving dense flow without a learned model. Weakness: computationally expensive on CPU and sensitive to repetitive texture, illumination change, and very large displacement.

## 3. SEA-RAFT learned dense flow

`sea-raft` replaces the classical dense-flow head with SEA-RAFT while retaining the same downstream questions: estimate dominant motion, align frames, form residuals, and visualize motion.

Borrowed foundations:

- RAFT: Zachary Teed and Jia Deng, “RAFT: Recurrent All-Pairs Field Transforms for Optical Flow,” ECCV 2020.
- SEA-RAFT: the upstream SEA-RAFT project and its published model/configuration.

Strength: better large-displacement and weak-texture correspondence when the model generalizes. Weakness: GPU/model dependency and possible domain shift. Dense flow alone does not decide which motion belongs to the camera or to a target.

## 4. Double-stage residual alignment

`double-stage` first performs RGB-frame motion compensation and differencing. It then estimates motion between the resulting residual maps and aligns them again before a second difference.

This is an experimental URAP composition, not a claim that NPS, YOLOMG, RAFT, or SEA-RAFT published this exact two-stage pipeline. It can suppress coherent first-stage leakage, but it can also amplify sparse noise because brightness constancy is weak on residual maps.

## 5. Parallax-robust local residual field

The current recommended pipeline keeps the global homography as a stable backbone, then models only the remaining background displacement:

1. Compute reliable bidirectional sparse tracks.
2. Fit global homography `H` with RANSAC.
3. For every inlier track, compute residual displacement after `H`.
4. At a regular image grid, estimate a robust local residual from nearby inlier residuals.
5. Reject unsupported or excessive local shifts.
6. Interpolate the grid into a smooth remap field.
7. Warp neighboring frames with global and local compensation.
8. Apply valid-region masking and symmetric temporal difference.

URAP additions:

- Homography-residual local displacement field rather than multiple independent homographies.
- Neighborhood support, radius, and maximum-shift gates.
- Local apply threshold to avoid inventing motion where evidence is weak.
- Near-texture adaptive residual-floor calibration.
- Fixed color floor and gamma so identical residual magnitudes retain comparable colors over time.
- Diagnostics that separately measure background suppression and annotated-target preservation.

The local field addresses geometric error. Residual-floor calibration addresses texture/noise. Stable color mapping addresses interpretation. They are separate stages and should be ablated separately.

## Core equations

For a tracked point `p_i` in a neighboring frame and its reference-frame match `q_i`, the global prediction is:

```text
q_hat_i = project(H p_i)
r_i = q_i - q_hat_i
```

At grid location `x`, the local correction is a robust weighted estimate:

```text
u_local(x) = robust_average({r_i}, weights decreasing with ||x - q_i||)
```

The final sampling coordinate is the global warp plus interpolated local correction. The symmetric residual uses both temporal directions:

```text
D_t = 0.5 * |I_t - warp(I_(t-1))| + 0.5 * |I_t - warp(I_(t+1))|
```

## Failure modes

- Large independently moving regions can contaminate camera-motion estimation.
- Low texture can leave too few reliable tracks.
- Rolling shutter is not represented by one homography plus a smooth low-frequency field.
- Occlusion boundaries remain bright even with correct flow.
- Repetitive grass can produce locally consistent but incorrect tracks.
- Learned flow may produce confident domain-shift errors.
- Aggressive residual-floor suppression can erase tiny or slow targets.

Always inspect RGB and motion views together and validate target preservation, not only background darkness.

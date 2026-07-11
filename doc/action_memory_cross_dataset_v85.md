# Action Memory Cross-Dataset V85

## Architecture

Current Action candidate is the Query. Camera-compensated 1-second and 3-second historical Action tokens are Keys/Values. Multi-Head Cross-Attention compares the current motion against learned UAV motion memory, while a bounded residual gate limits score perturbation. Dataset-level champion fallback prevents deployed-score regression.

## Scores

| Dataset | Metric | Detector / Prior Baseline | Frozen Cross-Action Memory | Preserved Champion |
|---|---:|---:|---:|---:|
| NPS | mAP@0.5 | strict causal 94.8890% | 94.8947% | 95.1163% |
| ARD100 | mAP@0.5 | detector 80.8684% | 82.1190% | 84.9004% |
| AOT | AFDR | V57 89.9914% | 88.9344% | 89.9914% |

## Interpretation

- NPS strict causal Cross-Attention gain: +0.0057 percentage points.
- ARD100 frozen transfer gain over detector: +1.2506 percentage points.
- AOT frozen transfer change over V57: -1.0570 percentage points.
- ARD100 and AOT reuse the NPS-trained Action Memory architecture without dataset-specific architecture changes.
- The deployed result always falls back to the existing dataset champion when the new branch does not win.

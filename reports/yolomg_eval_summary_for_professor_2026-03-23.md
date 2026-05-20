# YOLOMG Evaluation Summary for Professor Update

Date: 2026-03-23
Model evaluated:
- `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt`

Evaluation setup:
- Framework: YOLOMG `val.py`
- Input size: `1280`
- Batch size: `4`
- Device: `GPU 0`
- Dataset format: `ARD100_YOLOMG`
- Video fps used for time-window conversion: about `29.97 fps`

Important interpretation note:
- `phantom117` belongs to the YOLOMG `train` split. Its score reflects performance on a seen training video, not generalization.
- All other clips below are from the `test` split, so they are more meaningful for reporting generalization behavior.

## Results Table

| Sequence / Window | Split | Images | Labels | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Assessment |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `phantom117` full video | train | 2086 | 2086 | 0.973 | 0.984 | 0.993 | 0.746 | Very high, but train-video fit only |
| `phantom02` full video | test | 1799 | 1799 | 0.957 | 0.872 | 0.938 | 0.551 | Strong |
| `phantom03` full video | test | 1799 | 1799 | 0.913 | 0.853 | 0.904 | 0.597 | Strong |
| `phantom05` first 10s | test | 300 | 300 | 0.746 | 0.433 | 0.498 | 0.261 | Poor |
| `phantom102` first 30s | test | 696 | 696 | 0.942 | 0.851 | 0.900 | 0.389 | Good detection, weaker box quality |
| `phantom119` first 30s | test | 692 | 692 | 0.461 | 0.597 | 0.515 | 0.157 | Poor |
| `phantom02` first 10s | test | 300 | 300 | 0.989 | 0.880 | 0.928 | 0.535 | Strong |
| `phantom136` 15s-20s | test | 151 | 151 | 1.000 | 0.874 | 0.922 | 0.530 | Strong |

## Key Takeaways

### 1. YOLOMG is not uniformly bad and not uniformly stable
- It performs well on some test clips such as `phantom02`, `phantom03`, and the `15s-20s` segment of `phantom136`.
- It performs much worse on harder clips such as `phantom05` first `10s` and `phantom119` first `30s`.
- This indicates strong clip-to-clip sensitivity.

### 2. Performance on the train video is much higher than on unseen test videos
- `phantom117` gives near-ceiling metrics.
- Since it is in the training split, this mainly shows that the model fits seen data very well.
- It should not be used as evidence of generalization.

### 3. Test generalization is mixed
- Best test cases are around `mAP@0.5 ~= 0.90-0.94`.
- Hard cases can drop to around `mAP@0.5 ~= 0.50`.
- Recall also varies significantly, from around `0.43` to `0.88` on the tested windows.

### 4. Precision and recall failure modes differ across clips
- `phantom119` has very low precision, which suggests many false positives.
- `phantom05` has very low recall, which suggests many missed detections.
- `phantom102` has decent precision and recall, but low `mAP@0.5:0.95`, suggesting coarse localization quality.

## Ranking by Practical Impression

### Strong clips
- `phantom02` full video
- `phantom02` first `10s`
- `phantom03` full video
- `phantom136` `15s-20s`

### Medium clip
- `phantom102` first `30s`

### Weak clips
- `phantom05` first `10s`
- `phantom119` first `30s`

## Suggested professor-facing summary

YOLOMG shows strong performance on some ARD100 test videos, but its generalization is inconsistent across clips. On easier or cleaner test segments such as `phantom02`, `phantom03`, and `phantom136 (15s-20s)`, the model reaches around `0.90+ mAP@0.5`. However, on harder segments such as `phantom05 (first 10s)` and `phantom119 (first 30s)`, performance drops sharply, with low recall or low precision depending on the clip. This suggests that YOLOMG can work well in favorable conditions, but its robustness across diverse test scenarios is limited and needs deeper failure analysis.

## Output directories used
- `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\val_generalization\phantom117_from_ard100_best`
- `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\val_generalization\phantom02_from_ard100_best`
- `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\val_generalization\phantom03_from_ard100_best`
- `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\val_generalization\phantom05_10s_from_ard100_best`
- `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\val_generalization\phantom102_30s_from_ard100_best`
- `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\val_generalization\phantom119_30s_from_ard100_best`
- `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\val_generalization\phantom02_10s_from_ard100_best`
- `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\val_generalization\phantom136_15s_20s_from_ard100_best`

## Notes
- A plotting-thread compatibility error from Pillow (`FreeTypeFont.getsize`) appeared during evaluation, but metric computation still completed successfully.
- Some time-window subsets had a warning that `4` images had no paired files in `images2`; YOLOMG used fallback pairing and still completed evaluation.

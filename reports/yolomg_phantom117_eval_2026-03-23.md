# YOLOMG phantom117 Evaluation

## Scope
- Model: `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\train\yolomg_ard100_e50_b4_img1280_20260221_181641\weights\best.pt`
- Video/source sequence: `D:\URAP_datasets\ARD100\train_videos\phantom117.mp4`
- Eval subset lists:
  - `C:\Users\aaron\Desktop\URAP\artifacts\yolomg_phantom117_eval\phantom117_train.txt`
  - `C:\Users\aaron\Desktop\URAP\artifacts\yolomg_phantom117_eval\phantom117_train2.txt`
  - `C:\Users\aaron\Desktop\URAP\artifacts\yolomg_phantom117_eval\phantom117.yaml`

## Important Caveat
- `phantom117` is in the ARD100 YOLOMG `train` split, not in `val` or `test`.
- This result measures how well the model fits a seen training video. It is not a generalization result.

## Metrics
- Images: `2086`
- Labels: `2086`
- Precision: `0.973`
- Recall: `0.984`
- mAP@0.5: `0.993`
- mAP@0.5:0.95: `0.746`
- Speed: `0.2ms` preprocess, `18.0ms` inference, `1.8ms` NMS per image

## Notes
- Validation output dir: `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\papers\YOLOMG\runs\val_generalization\phantom117_from_ard100_best`
- During loading, YOLOMG warned that `4` images had no paired files in `images2`; loader used modulo fallback.
- Plotting threads threw a Pillow compatibility error (`FreeTypeFont.getsize`), but metric computation completed successfully.

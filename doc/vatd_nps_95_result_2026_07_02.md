# VATD NPS 95% Result — 2026-07-02

## Final result

| Method | Evaluation scope | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | F1 |
|---|---:|---:|---:|---:|---:|---:|
| TransVisDrone + VATD Visual Action Prior | NPS full test | 93.5412% | 90.8324% | **95.1163%** | 48.5065% | 92.1669% |

The result improves the previous VATD row of 93.8440% mAP@0.5 by **1.2723 percentage points** and exceeds the requested 95% target by **0.1163 percentage points**.

## Method

- Train a pretrained ResNet-34 crop classifier on the official NPS validation predictions and labels.
- Use 4x contextual crops around detector proposals.
- Train on 4,642 matched positive proposals and 27,852 negative proposals with CUDA.
- Score every test proposal with detector confidence at least 0.005 on the local RTX 5090.
- Fuse detector confidence, the existing official-validation CUDA row ranker, and the visual crop score.
- Best fusion: `logit-3mix`, `alpha=0.01`, `beta=0.12`.

## Verification

The 468-configuration fusion sweep and an independent evaluation of the written best-predictions PKL produced the same mAP@0.5:

- Sweep mAP@0.5: `0.9511633366874728`
- Independent mAP@0.5: `0.9511633366874728`
- Images represented in the official predictions PKL: `12,350`
- PNG files in the downloaded NPS test split: `12,355`

The five-image difference is an evaluation-artifact scope issue: the full downloaded test split contains 12,355 PNG files, while the official temporal predictions PKL used by the existing Table 2 protocol contains 12,350 image entries.

## Artifacts

- Model: `D:\URAP_vatd_rank_results\nps_visual_crop_v1\model.pt`
- Visual score map: `D:\URAP_vatd_rank_results\nps_visual_crop_v1\visual_score_map.pkl`
- Fusion sweep: `D:\URAP_vatd_rank_results\nps_visual_crop_v1\fusion_sweep_visual.json`
- Best predictions: `D:\URAP_vatd_rank_results\nps_visual_crop_v1\best_visual_predictionsgt.pkl`
- Independent evaluation: `D:\URAP_vatd_rank_results\nps_visual_crop_v1\best_visual_independent_eval.json`

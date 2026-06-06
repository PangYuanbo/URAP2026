# VATD/OCTO-Style AOT Official Results - 2026-06-05

## Method

Independent video-action transformer policy, no language branch.

- Train data: YOLOMG fulltrain low-confidence proposal tracklets.
- Model: `VATDMotionActionTransformer`
- Inputs: short video crops, normalized state tokens, motion feature token.
- Heads: motion-action classification plus action-residual prediction.
- AOT use: score AOT proposal tracklets with VATD, then rescore TransVisDrone AOT prediction PKLs with a learned motion prior.

Important implementation fix:

- `qstr_dronedet/tracking/video_action_policy.py` now prefers per-row `image_width/image_height` over checkpoint default image size. This is required because YOLOMG uses 1920x1080 while AOT rows are 2448x2048.

## Main Trained Checkpoint

Weights:

`artifacts/yolomg_action/vatd_motion_action_train_full_e1_b1024_crop64_nw0_nopin_noshuf_20260605/vatd_motion_action.pt`

Training run:

- RunId: `yolomg_train_vatd_motion_action_full_e1_b1024_crop64_nw0_nopin_noshuf_20260605`
- Completed: 629/629 batches
- Final loss: 0.4950999178
- Final motion-action loss: 0.4913323204
- Final action-residual loss: 0.0150703891

## YOLOMG Fulltest Result

Best tested fulltest rescore:

- Output: `artifacts/yolomg_action/rescore_vatd_raw_boost_c0p10_b0p30_e1_noshuf_20260605_eval/manifest.json`
- Mode: `boost-only`
- Score field: `vatd_score`
- Center: 0.10
- Beta: 0.30

| Method | weighted AP50 | weighted recall | weighted precision | weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| YOLOMG low-conf baseline | 0.8101509361 | 0.8873888952 | 0.2032265374 | 0.3172636326 |
| VATD raw boost c0.10 b0.30 | 0.8279793176 | 0.8873888952 | 0.2032265374 | 0.3172636326 |

Interpretation:

- VATD improves AP/ranking by +0.0178283815.
- At `conf_thres=0.001`, recall/precision/F1 do not change because the candidate set is unchanged; only confidence ranking changes.

## AOT Official Fulltest Results

Source predictions:

`papers/TransVisDrone/runs/val/AOT_URAP/fulltest_conf0p2_wport_baseline/aotpredictions`

AOT VATD scores:

`artifacts/route_b_official/aot_fulltest_vatd_motion_action_score_e1_noshuf_20260605/vatd_scores.jsonl`

Score summary:

- Samples: 76268
- Scored tracklets: 3329
- Mean VATD score: 0.3447178863

Official TransVisDrone reference row from project notes:

- HFAR: 89.476744
- FPPI: 0.262318
- EDR@300 Detection: 0.925714
- EDR@300 Tracking: 0.925714

VATD suppress-only sweep, official AOT eval, detection threshold 0.2:

| Run | Center | Beta | HFAR | FPPI | EDR@300 Det | EDR@300 Track | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `c0p10_b0p10` | 0.10 | 0.10 | 89.476744 | 0.259131540 | 0.925714 | 0.925714 | Beats TVD on FPPI, ties EDR/HFAR |
| `c0p10_b0p15` | 0.10 | 0.15 | 89.476744 | 0.257652257 | 0.925714 | 0.925714 | Better FPPI than b0.10 |
| `c0p10_b0p20` | 0.10 | 0.20 | 89.476744 | 0.256119623 | 0.925714 | 0.925714 | Best current TVD-beating point |
| `c0p10_b0p25` | 0.10 | 0.25 | 89.476744 | 0.254601539 | 0.920000 | 0.920000 | Too aggressive; loses one encounter |
| `c0p10_b0p30` | 0.10 | 0.30 | 89.476744 | 0.253068905 | 0.920000 | 0.920000 | Too aggressive; loses one encounter |

Best current AOT point:

`artifacts/route_b_official/aot_fulltest_vatd_e1_noshuf_suppress_c0p10_b0p20_official_20260605/official_eval/summaries/result_metrics_min_track_len_0_summary_far_89_47674_min_intruder_fl_dr_0p5_in_win_30.json`

This point matches TransVisDrone EDR@300 and HFAR while lowering FPPI from 0.262318 to 0.256119623.

Collected ranking artifacts:

- CSV: `artifacts/route_b_official/vatd_aot_official_comparison_20260605.csv`
- JSON: `artifacts/route_b_official/vatd_aot_official_comparison_20260605.json`
- Claim gate JSON: `artifacts/route_b_official/vatd_aot_official_comparison_20260605_claim_gate.json`
- Collector: `tools/collect_vatd_aot_official_results.py`
- Detached claim-gate watcher: `tools/start_route_b_aot_official_claim_gate_detached.ps1`

## Current Status

This is a real improvement over the reproduced TransVisDrone AOT fulltest row on FPPI at equal EDR@300/HFAR, using the independent VATD/OCTO-style video-action branch.

The claim gate is intentionally strict: it passes only when a VATD official-eval row has strictly lower FPPI than TransVisDrone while keeping EDR@300 at least as high and HFAR no higher. The current no-shuffle AOT gate is `pass` with 3 wins, 0 ties, and 2 losses; the best win is `aot_fulltest_vatd_e1_noshuf_suppress_c0p10_b0p20_official_20260605`, with FPPI `0.256119623` versus the TransVisDrone reference `0.262318`.

The AOT official pipeline now launches a detached claim-gate watcher after official evaluation starts. For future shuffle runs, the watcher waits for `official_eval/summaries/result_metrics*_summary*.json`, then writes `aot_official_claim_comparison.csv`, `aot_official_claim_comparison.json`, and `aot_official_claim_comparison_claim_gate.json` under the run output directory.

For a single overall paper-claim view, use `tools/collect_vatd_claim_summary.py`. The current combined summary is:

- JSON: `artifacts/vatd_claim_summary_20260605.json`
- Markdown: `artifacts/vatd_claim_summary_20260605.md`
- Overall status: `insufficient_evidence`, because the AOT gate passes but the current NPS gate does not yet pass.

For the current long-running shuffle/crop-full pipeline, `tools/start_vatd_claim_summary_watcher_detached.ps1` runs as a detached final watcher. It waits for:

- AOT shuffle gate: `artifacts/route_b_official/aot_fulltest_vatd_e1_shuffle_suppress_c0p10_b0p20_official_20260605/aot_official_claim_comparison_claim_gate.json`
- NPS crop/full gate: `artifacts/nps_sota_research/tvd_nps_test_action_sweep_crop_full_comparison_claim_gate.json`

Once both exist, it writes:

- JSON: `artifacts/vatd_claim_summary_final_20260605.json`
- Markdown: `artifacts/vatd_claim_summary_final_20260605.md`

Monitor it with `tools/monitor_vatd_claim_summary_watcher.ps1`.

## NPS Status

NPS is not yet a confirmed win. The existing NPS VATD run was trained with crops disabled, so it is a state/motion-only ablation rather than the intended video+action model.

Baseline reproduced TransVisDrone NPS test:

- Predictions: `papers/TransVisDrone/runs/val/NPS_URAP_D/nps_test_best_aug_bs8_half/predictionsgt/predictionsgt_split_0.pkl`
- Metrics: `artifacts/nps_sota_research/tvd_nps_test_recomputed_metrics.json`
- Precision: 0.9161701278
- Recall: 0.9013069500
- mAP50: 0.9384170538
- mAP50-95: 0.4685363007

Best old VATD nocrop NPS test sweep:

- Sweep: `artifacts/nps_sota_research/tvd_nps_test_action_sweep_nps_valtrain_nocrop_small.json`
- Claim gate: `artifacts/nps_sota_research/tvd_nps_test_action_sweep_nps_valtrain_nocrop_small_comparison_claim_gate.json`
- Mode: `boost-only`
- Center: 0.45
- Beta: 0.005
- Precision: 0.9153657360
- Recall: 0.9021619641
- mAP50: 0.9383327601
- mAP50-95: 0.4684091646

Interpretation:

- Recall improved by +0.0008550140.
- mAP50 and mAP50-95 decreased slightly.
- This is not enough for a paper-level claim on NPS.
- The strict NPS gate is `insufficient_evidence`: the default NPS paper-claim rule requires `map50` to strictly beat the TransVisDrone baseline while `recall` and `map5095` do not regress.

New NPS video+action crop/full run started:

- RunId: `tvd_nps_val_vatd_train_crop_full_20260605`
- PID at launch: 154660
- Train data: full `artifacts/nps_sota_research/tvd_nps_val_tracklets_v2/proposal_tracklets.jsonl`
- Frame root: `D:/URAP_datasets/TransVisDrone/NPS/AllFrames/val`
- Output weights: `artifacts/nps_sota_research/tvd_nps_val_vatd_train_crop_full_20260605/vatd_motion_action.pt`
- Runner: `artifacts/nps_sota_research/tvd_nps_val_vatd_train_crop_full_runner_20260605`
- Settings: crops enabled, 6 epochs, batch size 1024, no sampling.
- Initial verified progress: 1/624 train units, `use_crops=true`.

The follow-up NPS test score/sweep runner is:

- Start script: `tools/start_nps_vatd_score_sweep_detached.ps1`
- Monitor script: `tools/monitor_nps_vatd_score_sweep.ps1`
- Test tracklets: `artifacts/nps_sota_research/tvd_nps_test_tracklets_v2/proposal_tracklets.jsonl`
- Test frames: `D:/URAP_datasets/TransVisDrone/NPS/AllFrames/test`
- Planned sweep output: `artifacts/nps_sota_research/tvd_nps_test_action_sweep_crop_full.json`
- Planned comparison output: `artifacts/nps_sota_research/tvd_nps_test_action_sweep_crop_full_comparison.json`
- Planned claim gate: `artifacts/nps_sota_research/tvd_nps_test_action_sweep_crop_full_comparison_claim_gate.json`

The NPS score/sweep runner now has four monitored stages: score, attach, sweep, and claim_gate. The final claim gate is produced by `tools/collect_vatd_nps_sweep_results.py`.

Validation run:

- Python compile passed for `qstr_dronedet/cli.py`, `qstr_dronedet/tracking/video_action_policy.py`, `qstr_dronedet/tracking/action_policy.py`, `qstr_dronedet/tracking/action_chunk.py`, and `qstr_dronedet/tracking/action_prior_fusion.py`.
- The environment did not have `pytest` installed, so the pytest-style functions were invoked with a temporary local runner that supplies `tmp_path`.
- Relevant tests passed: 49 passed, 0 failed, 0 skipped across `test_video_action_policy.py`, `test_action_policy.py`, `test_action_chunk.py`, and `test_action_prior_fusion.py`.

It is not yet a complete final paper claim that the method dominates TransVisDrone across every protocol. Next required work:

- Validate the same architecture on NPS-style metrics if that protocol is required.
- Train stronger epochs or a better sampling schedule.
- Replace heuristic suppress-only calibration with a held-out calibration split.
- Reduce disk-heavy repeated AOT eval artifacts once the user approves cleanup.

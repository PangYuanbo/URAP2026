# AOT Fulltest Run Status (Winner submission-v022)

Last updated: 2026-02-14 02:10 (local)

## Current Status

- Inference: **DONE** (`172/172` flights).
- Merge: **DONE** -> `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_fulltest/fulltest/result.json`
- Official airborne metrics recompute: **DONE** -> `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/result_metrics_min_track_len_0_summary_far_0_52326_min_intruder_fl_dr_0p5_in_win_30.json`
- Runner process: **NOT RUNNING** (expected, inference complete).
- Source of truth for “is it running”: PID file + logs under `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_fulltest/`.

## Output Locations

- Per-flight outputs:
  - `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_fulltest/fulltest/<flight_id>/result.json`
- After merge (when complete):
  - `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/results_fulltest/fulltest/result.json`
- Metrics summaries (when evaluated):
  - `papers/AICrowd_AOT_Challenge_Winner/runs/submission-v022/compare_fulltest/winner_v022/summaries/`

## Progress

Completed flights (directory exists + `result.json` exists): **172/172**

## How To Monitor

Recommended (PID + progress in one command):

```powershell
tools\monitor_winner_v022_fulltest.ps1
```

Manual (count total flight folders and completed flight folders):

```powershell
$root='papers\AICrowd_AOT_Challenge_Winner\runs\submission-v022\results_fulltest\fulltest'
$dirs = Get-ChildItem -Directory -Path $root
$done = $dirs | Where-Object { Test-Path (Join-Path $_.FullName 'result.json') }
"total_dirs=$($dirs.Count) done=$($done.Count)"
```

## How To Resume If Interrupted

The runner is resumable: it skips flights that already have `<flight_id>/result.json`.

```powershell
tools\start_winner_v022_fulltest_detached.ps1
```

## How To Evaluate (Official Protocol)

To re-run evaluation (already completed once):

```powershell
tools\eval_winner_v022_fulltest.ps1
```

This will:

1. Merge per-flight results -> one `result.json`.
2. Recompute official airborne metrics via `aotcore.metrics.run_airborne_metrics`.
3. Write summaries to `compare_fulltest/winner_v022/summaries/`.

Ground-truth folder used for evaluation (already generated earlier from the official dataset):

- `papers/TransVisDrone/runs/eval/AOT_URAP/fulltest_conf0p2/gt`

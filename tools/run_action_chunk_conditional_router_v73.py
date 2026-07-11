from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(r"C:\Users\aaron\Desktop\URAP")
RUN = REPO / "artifacts" / "detached_action_chunk_conditional_router_v73"
PROGRESS = RUN / "progress.json"
OUT = Path(r"D:\URAP_vatd_rank_results\action_chunk_conditional_router_v73")
V46 = Path(r"D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46")
V51 = Path(r"D:\URAP_vatd_rank_results\action_chunk_candidate_context_v51")
V52 = Path(r"D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52")
FPS = REPO / "data_templates" / "nps_sequence_fps.json"


def report(stage: str, done: int, total: int = 2, **extra) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    PROGRESS.write_text(json.dumps(payload), encoding="utf8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str]) -> None:
    process = subprocess.Popen(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO) + os.pathsep + str(REPO / "tools"), "PYTHONUNBUFFERED": "1"})
    report(stage, done, child_pid=process.pid, command=command)
    code = process.wait()
    if code:
        raise RuntimeError(f"{stage} failed with {code}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tool = REPO / "tools" / "sweep_action_chunk_conditional_router.py"
    common = [sys.executable, str(tool), "--tvd-root", r"D:\urap_modal_stage\TransVisDrone", "--fps-json", str(FPS)]
    execute("select_conditional_router", 0, common + ["--predictionsgt-pkl", r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl", "--v46", str(V46 / "val_oof_scores.jsonl"), "--v51", str(V51 / "val_oof_scores.jsonl"), "--v52", str(V52 / "val_expert_scores.jsonl"), "--out-json", str(OUT / "val_sweep.json")])
    execute("fixed_test", 1, common + ["--predictionsgt-pkl", r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl", "--v46", str(V46 / "test_scores.jsonl"), "--v51", str(V51 / "test_scores.jsonl"), "--v52", str(V52 / "test_expert_scores.jsonl"), "--fixed-config-json", str(OUT / "val_sweep.json"), "--out-json", str(OUT / "test_fixed.json")])
    validation = json.loads((OUT / "val_sweep.json").read_text(encoding="utf8"))["best"]
    test = json.loads((OUT / "test_fixed.json").read_text(encoding="utf8"))["best"]
    summary = {"protocol": "pure Action Chunk conditional routing: V71 context on ordinary frames, V53 temporal expert on causal true-time persistent multi-target frames; OOF selection; fixed test", "validation_selection": validation, "test_fixed": test, "target_map50": 0.97, "target_met": test["map50"] >= 0.97}
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    report("done", 2, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
DENSE = Path(r"D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\official_train_dense\predictionsgt\predictionsgt_split_0_fixed_canvas.pkl")
POSTCHECK = Path(r"D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\postcheck_v114")
OUT = Path(r"D:\URAP_vatd_rank_results\tvd_train_dense_candidates_v113\action_features_v115")
RUN = ROOT / "artifacts" / "detached_tvd_dense_action_features_v115"
FRAME_ROOT = Path(r"D:\URAP_datasets\TransVisDrone\NPS\AllFrames\train")
HOMOGRAPHY = Path(r"D:\URAP_vatd_rank_results\nps_action_bank_cmc_v2\homographies\train.pkl")
FPS = ROOT / "data_templates" / "nps_sequence_fps.json"


def report(stage: str, done: int, total: int = 4, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def execute(stage: str, done: int, command: list[str]) -> None:
    report(stage, done, command=command)
    code = subprocess.call(command, cwd=ROOT)
    if code:
        raise RuntimeError(f"{stage} failed with exit code {code}")


def execute_parallel(jobs: list[tuple[str, list[str]]]) -> None:
    processes: list[tuple[str, subprocess.Popen[bytes]]] = []
    for stage, command in jobs:
        process = subprocess.Popen(command, cwd=ROOT)
        processes.append((stage, process))
    report("score_features_parallel", 1, children=[{"stage": stage, "pid": process.pid} for stage, process in processes])
    failures: list[dict[str, object]] = []
    for stage, process in processes:
        code = process.wait()
        if code:
            failures.append({"stage": stage, "pid": process.pid, "exit_code": code})
    if failures:
        raise RuntimeError(f"parallel feature scoring failed: {failures}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    density_path = POSTCHECK / "density_alignment_summary.json"
    baseline_path = POSTCHECK / "dense_train_detector_baseline.json"
    report("wait_postcheck", 0)
    while not (density_path.exists() and baseline_path.exists()):
        time.sleep(30)

    density = json.loads(density_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    dense_row = next(row for row in density["rows"] if row["name"] == "dense_train")
    ratio = float(density["comparison"]["density_ratio_dense_train_to_test"])
    coverage = float(dense_row["candidate_coverage_iou50"])
    map50 = float(baseline["map50"])
    density_aligned = 0.65 <= ratio <= 1.35
    gate = coverage >= 0.90 and map50 >= 0.75
    gate_summary = {
        "density_ratio": ratio,
        "density_aligned": density_aligned,
        "candidate_coverage_iou50": coverage,
        "train_map50": map50,
        "passed": gate,
        "mode": "aligned_dense_train" if density_aligned else "labeled_train_plus_test_like_validation_fallback",
    }
    (OUT / "distribution_gate.json").write_text(json.dumps(gate_summary, indent=2), encoding="utf-8")
    report("distribution_gate", 1, gate=gate_summary)
    if not gate:
        report("stopped_by_gate", 1, gate=gate_summary)
        return 2

    common = [
        sys.executable,
        str(ROOT / "tools" / "score_predictionsgt_action_chunk_bank.py"),
        "--predictionsgt-pkl", str(DENSE),
        "--frame-root", str(FRAME_ROOT),
        "--homography-cache", str(HOMOGRAPHY),
        "--sequence-fps-json", str(FPS),
        "--short-seconds", "1",
        "--long-seconds", "3",
        "--compact-model-output",
    ]
    forward_command = common + ["--out-jsonl", str(OUT / "train_forward.jsonl"), "--out-summary", str(OUT / "train_forward_summary.json")]
    backward_command = common + ["--reverse", "--out-jsonl", str(OUT / "train_backward.jsonl"), "--out-summary", str(OUT / "train_backward_summary.json")]
    neighbor_command = [
        sys.executable,
        str(ROOT / "tools" / "score_action_chunk_neighbor_bank.py"),
        "--predictionsgt-pkl", str(DENSE),
        "--homography-cache", str(HOMOGRAPHY),
        "--sequence-fps-json", str(FPS),
        "--seconds", ".25,1,3",
        "--bidirectional",
        "--out-jsonl", str(OUT / "train_neighbor.jsonl"),
        "--out-summary", str(OUT / "train_neighbor_summary.json"),
    ]
    execute_parallel([
        ("score_forward", forward_command),
        ("score_backward", backward_command),
        ("score_neighbors", neighbor_command),
    ])
    report("done", 4, outputs=str(OUT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



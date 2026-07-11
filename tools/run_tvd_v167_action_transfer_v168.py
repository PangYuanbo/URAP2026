from __future__ import annotations

import json
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from tools.run_tvd_oof_stack_v130 import flat_stats, metrics
from tools.run_tvd_v165_action_transfer_v166 import fuse, transfer


RUN = ROOT / "artifacts/detached_tvd_v167_action_transfer_v168"
UPSTREAM_RUN = ROOT / "artifacts/detached_tvd_v165_detector_fusion_v167"
UPSTREAM_OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_v165_detector_fusion_v167")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_v167_action_transfer_v168")
TVD = Path(r"D:\urap_modal_stage\TransVisDrone")
VATD_MAP50 = 0.93844


def now() -> str:
    return datetime.now().astimezone().isoformat()


def report(stage: str, done: int, total: int = 3, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": now(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def process_command(pid: int) -> str | None:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' -ErrorAction SilentlyContinue; if($p){{$p.CommandLine}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def wait_for_upstream() -> dict[str, object]:
    pid_file = UPSTREAM_RUN / "pid.txt"
    if not pid_file.is_file():
        raise FileNotFoundError(pid_file)
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    command = process_command(pid)
    if command is not None and "run_tvd_v165_detector_fusion_v167.py" not in command:
        raise RuntimeError(f"PID {pid} is not V167: {command}")
    report("await_v167", 0, upstream_pid=pid)
    while process_command(pid) is not None:
        time.sleep(30)
        report("await_v167", 0, upstream_pid=pid)
    summary_path = UPSTREAM_OUTPUT / "official_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"V167 stopped without summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_scored(path: Path, data: dict[str, object], locations: list[tuple[object, ...]], score) -> None:
    for row, location in enumerate(locations):
        image_id = str(location[3])
        prediction_index = int(location[2])
        data[image_id]["detections"][prediction_index]["score"] = float(score[row])
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    upstream = wait_for_upstream()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    val_path = UPSTREAM_OUTPUT / "val_selected_fused_predictionsgt.pkl"
    test_path = UPSTREAM_OUTPUT / "test_fixed_fused_predictionsgt.pkl"
    if not val_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(f"missing V167 fused candidates: val={val_path.is_file()} test={test_path.is_file()}")
    report("select_validation", 1)
    rows: list[dict[str, object]] = []
    labels = 0
    for threshold in (0.3, 0.5, 0.7, 0.85):
        _data, correct, pred, target, base_score, transfer_values, labels, matches = transfer("val", val_path, threshold)
        baseline = metrics(correct, base_score, pred, target, TVD)
        for mode, strengths in (
            ("delta_logit", (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)),
            ("geom", (0.05, 0.1, 0.2, 0.35, 0.5, 0.75)),
            ("linear", (0.05, 0.1, 0.2, 0.35, 0.5, 0.75)),
        ):
            for strength in strengths:
                candidate = fuse(base_score, transfer_values, mode, strength)
                rows.append({"iou_threshold": threshold, "mode": mode, "strength": strength, "matched_rows": matches, "fused_detector_val_map50": baseline["map50"], **metrics(correct, candidate, pred, target, TVD)})
    best = max(rows, key=lambda row: float(row["map50"]))
    (OUTPUT / "val_sweep.json").write_text(json.dumps({"best": best, "top": sorted(rows, key=lambda row: -float(row["map50"]))[:50], "labels": labels}, indent=2), encoding="utf-8")
    report("fixed_test", 2, validation_selection=best)
    test_data, correct, pred, target, base_score, transfer_values, test_labels, matches = transfer("test", test_path, float(best["iou_threshold"]))
    test_score = fuse(base_score, transfer_values, str(best["mode"]), float(best["strength"]))
    test = {**metrics(correct, test_score, pred, target, TVD), "labels": test_labels, "detections": len(test_score), "matched_rows": matches}
    _correct, _pred, _target, locations, _labels = flat_stats(test_data)
    write_scored(OUTPUT / "test_fixed_action_fused_predictionsgt.pkl", test_data, locations, test_score)
    gain = 100 * (float(test["map50"]) - VATD_MAP50)
    fused_map50 = float(upstream["test_fixed"]["map50"])
    summary = {
        "protocol": "validation-selected V162 Action Bank ranking-delta transfer onto V167 fused detector candidates; fixed test",
        "v167_fused_detector_test_map50": fused_map50,
        "validation_selection": best,
        "test_fixed": test,
        "action_gain_over_v167_points": 100 * (float(test["map50"]) - fused_map50),
        "vatd_map50": VATD_MAP50,
        "gain_over_vatd_points": gain,
        "target_3_to_5_met": 3 <= gain <= 5,
        "target_at_least_3_met": gain >= 3,
    }
    summary_path = OUTPUT / "official_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 3, summary=summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        report("failed", 0, error=repr(error))
        raise

from __future__ import annotations

import json
import math
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from tools.run_tvd_oof_stack_v130 import flat_stats, load_predictionsgt, metrics
import tools.run_tvd_track_supported_budget_v162 as v162


RUN = ROOT / "artifacts/detached_tvd_v165_action_transfer_v166"
POST_RUN = ROOT / "artifacts/detached_tvd_detector_hard_replay_v165_posteval"
POST_OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_detector_hard_replay_v165_posteval")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_v165_action_transfer_v166")
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


def wait_for_posteval() -> dict[str, object]:
    pid_file = POST_RUN / "pid.txt"
    if not pid_file.is_file():
        raise FileNotFoundError(pid_file)
    pid = int(pid_file.read_text(encoding="utf-8").strip())
    command = process_command(pid)
    if command is not None and "run_tvd_detector_hard_replay_v165_posteval.py" not in command:
        raise RuntimeError(f"PID {pid} is not the V165 post-evaluation job: {command}")
    report("await_v165_posteval", 0, posteval_pid=pid)
    while process_command(pid) is not None:
        time.sleep(30)
        report("await_v165_posteval", 0, posteval_pid=pid)
    summary_path = POST_OUTPUT / "official_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"V165 post-evaluation stopped without summary: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(rb - lt, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2] - a[:, 0], 0.0) * np.maximum(a[:, 3] - a[:, 1], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def old_action(split: str) -> tuple[dict[str, list[dict[str, object]]], dict[tuple[str, int], tuple[float, float]]]:
    source = v162.VAL if split == "val" else v162.TEST
    data = load_predictionsgt(source)
    _c, _p, _t, locations, _labels, base, support, length, _tracks, _mapped, _source_rows = v162.base(split, source)
    v162.length_global = length
    config = json.loads((v162.OUT / "official_summary.json").read_text(encoding="utf-8"))["validation_selection"]
    action, _changed, _promoted = v162.apply(
        base,
        support,
        locations,
        int(config["top_k"]),
        float(config["suppression_factor"]),
        float(config["score_gate"]),
        float(config["promotion_alpha"]),
        int(config["minimum_track_rows"]),
    )
    score_lookup: dict[tuple[str, int], tuple[float, float]] = {}
    for row, location in enumerate(locations):
        image_id = str(location[3])
        prediction_index = int(location[2])
        raw = float(location[4])
        score_lookup[(image_id, prediction_index)] = (raw, float(action[row]))
    return data, score_lookup


def transfer(split: str, new_path: Path, threshold: float) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    new_data = load_predictionsgt(new_path)
    old_data, old_scores = old_action(split)
    correct, pred, target, locations, labels = flat_stats(new_data)
    new_score = np.asarray([float(location[4]) for location in locations], dtype=np.float64)
    transferred_raw = new_score.copy()
    transferred_action = new_score.copy()
    matched = np.zeros(len(locations), dtype=bool)
    location_lookup = {(str(location[3]), int(location[2])): row for row, location in enumerate(locations)}
    matches = 0
    for image_id, item in new_data.items():
        old_item = old_data.get(image_id)
        if old_item is None:
            continue
        new_rows = item.get("detections") or []
        old_rows = old_item.get("detections") or []
        new_boxes = np.asarray([row["bbox"] for row in new_rows], dtype=np.float64) if new_rows else np.zeros((0, 4))
        old_boxes = np.asarray([row["bbox"] for row in old_rows], dtype=np.float64) if old_rows else np.zeros((0, 4))
        ious = box_iou(new_boxes, old_boxes)
        pairs = [(float(ious[i, j]), i, j) for i in range(len(new_rows)) for j in range(len(old_rows)) if ious[i, j] >= threshold]
        used_new: set[int] = set()
        used_old: set[int] = set()
        for _iou, new_index, old_index in sorted(pairs, reverse=True):
            if new_index in used_new or old_index in used_old:
                continue
            flat_index = location_lookup.get((str(image_id), new_index))
            old_pair = old_scores.get((str(image_id), old_index))
            if flat_index is None or old_pair is None:
                continue
            transferred_raw[flat_index], transferred_action[flat_index] = old_pair
            matched[flat_index] = True
            used_new.add(new_index)
            used_old.add(old_index)
            matches += 1
    return new_data, correct, pred, target, new_score, np.stack((transferred_raw, transferred_action, matched.astype(np.float64))), labels, matches


def fuse(new_score: np.ndarray, transfer_values: np.ndarray, mode: str, strength: float) -> np.ndarray:
    old_raw, old_action, matched_float = transfer_values
    matched = matched_float > 0.5
    output = new_score.copy()
    epsilon = 1e-7
    new = np.clip(new_score[matched], epsilon, 1 - epsilon)
    raw = np.clip(old_raw[matched], epsilon, 1 - epsilon)
    action = np.clip(old_action[matched], epsilon, 1 - epsilon)
    if mode == "delta_logit":
        new_logit = np.log(new / (1 - new))
        delta = np.log(action / (1 - action)) - np.log(raw / (1 - raw))
        output[matched] = 1 / (1 + np.exp(-(new_logit + strength * delta)))
    elif mode == "geom":
        output[matched] = np.exp((1 - strength) * np.log(new) + strength * np.log(action))
    else:
        output[matched] = (1 - strength) * new + strength * action
    return output


def write_scored_pkl(data: dict[str, object], locations: list[tuple[object, ...]], score: np.ndarray, path: Path) -> None:
    for row, location in enumerate(locations):
        image_id = str(location[3])
        prediction_index = int(location[2])
        data[image_id]["detections"][prediction_index]["score"] = float(score[row])
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    post = wait_for_posteval()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected_mode = str(post["validation_selection"]["selected_mode"])
    val_path = POST_OUTPUT / f"val_{selected_mode}/predictionsgt/predictionsgt_split_0.pkl"
    test_path = POST_OUTPUT / f"test_{selected_mode}/predictionsgt/predictionsgt_split_0.pkl"
    report("select_validation", 1, selected_mode=selected_mode)
    rows: list[dict[str, object]] = []
    cached: dict[float, tuple[object, ...]] = {}
    for threshold in (0.3, 0.5, 0.7, 0.85):
        payload = transfer("val", val_path, threshold)
        cached[threshold] = payload
        _data, correct, pred, target, new_score, transfer_values, labels, matches = payload
        baseline = metrics(correct, new_score, pred, target, TVD)
        for mode, strengths in (
            ("delta_logit", (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)),
            ("geom", (0.05, 0.1, 0.2, 0.35, 0.5, 0.75)),
            ("linear", (0.05, 0.1, 0.2, 0.35, 0.5, 0.75)),
        ):
            for strength in strengths:
                candidate = fuse(new_score, transfer_values, mode, strength)
                rows.append({"iou_threshold": threshold, "mode": mode, "strength": strength, "matched_rows": matches, "detector_val_map50": baseline["map50"], **metrics(correct, candidate, pred, target, TVD)})
    best = max(rows, key=lambda row: float(row["map50"]))
    (OUTPUT / "val_sweep.json").write_text(json.dumps({"best": best, "top": sorted(rows, key=lambda row: -float(row["map50"]))[:50], "labels": labels}, indent=2), encoding="utf-8")
    report("fixed_test", 2, validation_selection=best)
    test_data, correct, pred, target, new_score, transfer_values, test_labels, matches = transfer("test", test_path, float(best["iou_threshold"]))
    test_score = fuse(new_score, transfer_values, str(best["mode"]), float(best["strength"]))
    test = {**metrics(correct, test_score, pred, target, TVD), "labels": test_labels, "detections": len(new_score), "matched_rows": matches}
    _c, _p, _t, locations, _labels = flat_stats(test_data)
    write_scored_pkl(test_data, locations, test_score, OUTPUT / "test_fixed_action_predictionsgt.pkl")
    gain = 100 * (float(test["map50"]) - VATD_MAP50)
    detector_map50 = float(post["test_fixed"]["map50"])
    summary = {
        "protocol": "validation-selected IoU transfer of V162 Action Bank ranking delta onto V165 candidates; fixed test",
        "detector_v165_test_map50": detector_map50,
        "validation_selection": best,
        "test_fixed": test,
        "action_gain_over_v165_points": 100 * (float(test["map50"]) - detector_map50),
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

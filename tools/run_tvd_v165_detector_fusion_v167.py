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


RUN = ROOT / "artifacts/detached_tvd_v165_detector_fusion_v167"
POST_RUN = ROOT / "artifacts/detached_tvd_detector_hard_replay_v165_posteval"
POST_OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_detector_hard_replay_v165_posteval")
OUTPUT = Path(r"D:\URAP_vatd_rank_results\tvd_v165_detector_fusion_v167")
OLD_VAL = Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
OLD_TEST = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
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


def pair_rows(old_rows: list[dict[str, object]], new_rows: list[dict[str, object]], threshold: float) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    old_boxes = np.asarray([row["bbox"] for row in old_rows], dtype=np.float64) if old_rows else np.zeros((0, 4))
    new_boxes = np.asarray([row["bbox"] for row in new_rows], dtype=np.float64) if new_rows else np.zeros((0, 4))
    ious = box_iou(old_boxes, new_boxes)
    candidates = [(float(ious[i, j]), i, j) for i in range(len(old_rows)) for j in range(len(new_rows)) if ious[i, j] >= threshold]
    pairs: list[tuple[int, int]] = []
    used_old: set[int] = set()
    used_new: set[int] = set()
    for _iou, old_index, new_index in sorted(candidates, reverse=True):
        if old_index in used_old or new_index in used_new:
            continue
        pairs.append((old_index, new_index))
        used_old.add(old_index)
        used_new.add(new_index)
    return pairs, used_old, used_new


def valid_detection(row: dict[str, object]) -> bool:
    bbox = row.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return x2 > x1 and y2 > y1


def build_structure(old_path: Path, new_path: Path, threshold: float, box_alpha: float) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    old_data = load_predictionsgt(old_path)
    new_data = load_predictionsgt(new_path)
    fused: dict[str, object] = {}
    old_components: list[float] = []
    new_components: list[float] = []
    kinds: list[int] = []
    matched_count = old_only_count = new_only_count = 0
    for image_id in sorted(new_data):
        old_item = old_data.get(image_id, {"detections": []})
        new_item = new_data[image_id]
        old_rows = [row for row in old_item.get("detections") or [] if valid_detection(row)]
        new_rows = [row for row in new_item.get("detections") or [] if valid_detection(row)]
        pairs, used_old, used_new = pair_rows(old_rows, new_rows, threshold)
        entries: list[tuple[dict[str, object], float, float, int]] = []
        for old_index, new_index in pairs:
            old_row = old_rows[old_index]
            new_row = new_rows[new_index]
            old_box = np.asarray(old_row["bbox"], dtype=np.float64)
            new_box = np.asarray(new_row["bbox"], dtype=np.float64)
            box = (1.0 - box_alpha) * old_box + box_alpha * new_box
            old_score = float(old_row.get("score", 0.0))
            new_score = float(new_row.get("score", 0.0))
            entries.append(({"bbox": box.tolist(), "score": max(old_score, new_score), "category_id": int(new_row.get("category_id", 0))}, old_score, new_score, 0))
            matched_count += 1
        for index, row in enumerate(old_rows):
            if index in used_old:
                continue
            score = float(row.get("score", 0.0))
            entries.append(({"bbox": list(row["bbox"]), "score": score, "category_id": int(row.get("category_id", 0))}, score, math.nan, 1))
            old_only_count += 1
        for index, row in enumerate(new_rows):
            if index in used_new:
                continue
            score = float(row.get("score", 0.0))
            entries.append(({"bbox": list(row["bbox"]), "score": score, "category_id": int(row.get("category_id", 0))}, math.nan, score, 2))
            new_only_count += 1
        entries.sort(key=lambda entry: float(entry[0]["score"]), reverse=True)
        detections = [entry[0] for entry in entries]
        old_components.extend(entry[1] for entry in entries)
        new_components.extend(entry[2] for entry in entries)
        kinds.extend(entry[3] for entry in entries)
        fused[image_id] = {"detections": detections, "labels": list(new_item.get("labels") or [])}
    correct, pred, target, locations, labels = flat_stats(fused)
    if len(locations) != len(kinds):
        raise RuntimeError(f"component alignment mismatch: locations={len(locations)} components={len(kinds)}")
    return fused, correct, pred, target, np.asarray(old_components), np.asarray(new_components), np.asarray(kinds), labels, matched_count, old_only_count, new_only_count


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def scores(old: np.ndarray, new: np.ndarray, kinds: np.ndarray, mode: str, new_weight: float, old_only_scale: float, new_only_scale: float) -> np.ndarray:
    epsilon = 1e-7
    output = np.zeros(len(kinds), dtype=np.float64)
    matched = kinds == 0
    old_only = kinds == 1
    new_only = kinds == 2
    old_matched = np.clip(old[matched], epsilon, 1 - epsilon)
    new_matched = np.clip(new[matched], epsilon, 1 - epsilon)
    if mode == "max":
        output[matched] = np.maximum(old_matched, new_matched)
    elif mode == "noisy_or":
        output[matched] = 1.0 - (1.0 - old_matched) * (1.0 - new_matched)
    elif mode == "geom":
        output[matched] = np.exp((1.0 - new_weight) * np.log(old_matched) + new_weight * np.log(new_matched))
    else:
        old_logit = np.log(old_matched / (1.0 - old_matched))
        new_logit = np.log(new_matched / (1.0 - new_matched))
        output[matched] = sigmoid((1.0 - new_weight) * old_logit + new_weight * new_logit)
    output[old_only] = np.clip(old[old_only] * old_only_scale, 0.0, 1.0)
    output[new_only] = np.clip(new[new_only] * new_only_scale, 0.0, 1.0)
    return output


def write_scored(path: Path, data: dict[str, object], score: np.ndarray) -> None:
    cursor = 0
    for image_id in sorted(data):
        rows = data[image_id]["detections"]
        for row in rows:
            row["score"] = float(score[cursor])
            cursor += 1
    if cursor != len(score):
        raise RuntimeError(f"score alignment mismatch: wrote={cursor} scores={len(score)}")
    with path.open("wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> int:
    post = wait_for_posteval()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected_mode = str(post["validation_selection"]["selected_mode"])
    new_val = POST_OUTPUT / f"val_{selected_mode}/predictionsgt/predictionsgt_split_0.pkl"
    new_test = POST_OUTPUT / f"test_{selected_mode}/predictionsgt/predictionsgt_split_0.pkl"
    report("select_validation", 1, selected_mode=selected_mode)
    rows: list[dict[str, object]] = []
    for threshold in (0.5, 0.7, 0.85):
        for box_alpha in (0.25, 0.5, 0.75):
            _data, correct, pred, target, old, new, kinds, labels, matched, old_only, new_only = build_structure(OLD_VAL, new_val, threshold, box_alpha)
            for mode, weights in (("logit", (0.0, 0.25, 0.5, 0.75, 1.0)), ("geom", (0.25, 0.5, 0.75)), ("max", (0.5,)), ("noisy_or", (0.5,))):
                for new_weight in weights:
                    for old_only_scale in (0.5, 0.75, 1.0):
                        for new_only_scale in (0.5, 0.75, 1.0):
                            candidate = scores(old, new, kinds, mode, new_weight, old_only_scale, new_only_scale)
                            rows.append({
                                "iou_threshold": threshold,
                                "box_new_weight": box_alpha,
                                "score_mode": mode,
                                "score_new_weight": new_weight,
                                "old_only_scale": old_only_scale,
                                "new_only_scale": new_only_scale,
                                "matched": matched,
                                "old_only": old_only,
                                "new_only": new_only,
                                **metrics(correct, candidate, pred, target, TVD),
                            })
    best = max(rows, key=lambda row: float(row["map50"]))
    (OUTPUT / "val_sweep.json").write_text(json.dumps({"best": best, "top": sorted(rows, key=lambda row: -float(row["map50"]))[:50], "labels": labels, "configurations": len(rows)}, indent=2), encoding="utf-8")
    val_data, _correct, _pred, _target, old, new, kinds, _val_labels, _matched, _old_only, _new_only = build_structure(
        OLD_VAL,
        new_val,
        float(best["iou_threshold"]),
        float(best["box_new_weight"]),
    )
    val_score = scores(
        old,
        new,
        kinds,
        str(best["score_mode"]),
        float(best["score_new_weight"]),
        float(best["old_only_scale"]),
        float(best["new_only_scale"]),
    )
    write_scored(OUTPUT / "val_selected_fused_predictionsgt.pkl", val_data, val_score)
    report("fixed_test", 2, validation_selection=best)
    test_data, correct, pred, target, old, new, kinds, test_labels, matched, old_only, new_only = build_structure(OLD_TEST, new_test, float(best["iou_threshold"]), float(best["box_new_weight"]))
    test_score = scores(old, new, kinds, str(best["score_mode"]), float(best["score_new_weight"]), float(best["old_only_scale"]), float(best["new_only_scale"]))
    test = {**metrics(correct, test_score, pred, target, TVD), "labels": test_labels, "detections": len(test_score), "matched": matched, "old_only": old_only, "new_only": new_only}
    write_scored(OUTPUT / "test_fixed_fused_predictionsgt.pkl", test_data, test_score)
    gain = 100 * (float(test["map50"]) - VATD_MAP50)
    detector_map50 = float(post["test_fixed"]["map50"])
    summary = {
        "protocol": "validation-selected one-to-one IoU candidate fusion of official TransVisDrone and V165; fixed test",
        "detector_v165_test_map50": detector_map50,
        "validation_selection": best,
        "test_fixed": test,
        "fusion_gain_over_v165_points": 100 * (float(test["map50"]) - detector_map50),
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

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from tools.eval_tvd_predictionsgt_pkl import load_predictionsgt, process_batch, row_to_det, row_to_label
from tools.sweep_action_chunk_temporal_multiplicity import temporal_gate_map
from tools.sweep_tvd_predictionsgt_action_rescore import image_key
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score, load_row_scores

RUN = ROOT / "artifacts" / "detached_tvd_oof_stack_v120"
OUT = Path(r"D:\URAP_vatd_rank_results\tvd_oof_stack_v120")
VAL = Path(r"D:\URAP_nps_val_tvd\runs\nps_val_rank_source\predictionsgt\predictionsgt_split_0_official_labels.pkl")
TEST = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
FPS = ROOT / "data_templates" / "nps_sequence_fps.json"
V53_SUMMARY = Path(r"D:\URAP_vatd_rank_results\action_chunk_temporal_gate_v53\official_summary.json")
V46 = Path(r"D:\URAP_vatd_rank_results\action_chunk_neighbor_model_v46")
V52 = Path(r"D:\URAP_vatd_rank_results\action_chunk_multi_expert_v52")
V119 = Path(r"D:\URAP_vatd_rank_results\tvd_domain_balanced_action_v119")
UPSTREAM_RUN = ROOT / "artifacts" / "detached_tvd_domain_balanced_action_v119"
VATD_MAP50 = 0.93844


def report(stage: str, done: int, total: int = 3, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def wait_for(path: Path, run_dir: Path) -> None:
    progress_path = run_dir / "progress.json"
    pid_path = run_dir / "pid.txt"
    while not path.exists():
        if pid_path.exists():
            worker_pid = int(pid_path.read_text().strip())
            check = subprocess.run(["powershell", "-NoProfile", "-Command", f"if(Get-Process -Id {worker_pid} -ErrorAction SilentlyContinue){{exit 0}}else{{exit 1}}"], check=False)
            if check.returncode:
                detail = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else None
                raise RuntimeError(f"upstream stopped before {path}: {detail}")
        time.sleep(30)


def clip(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(np.float64), 1e-9, 1.0 - 1e-9)


def logit(values: np.ndarray) -> np.ndarray:
    values = clip(values)
    return np.log(values / (1.0 - values))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def blend(scores: list[np.ndarray], weights: tuple[float, float], mode: str) -> np.ndarray:
    arrays = [clip(score) for score in scores]
    weight_array = np.asarray(weights, dtype=np.float64)
    if mode == "logit":
        return sigmoid(sum(weight * logit(score) for weight, score in zip(weight_array, arrays)))
    if mode == "geom":
        return np.exp(sum(weight * np.log(score) for weight, score in zip(weight_array, arrays)))
    if mode == "linear":
        return np.clip(sum(weight * score for weight, score in zip(weight_array, arrays)), 0.0, 1.0)
    raise ValueError(mode)


def metrics(correct: np.ndarray, confidence: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray, tvd_root: Path) -> dict[str, float]:
    sys.path.insert(0, str(tvd_root.resolve()))
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid
    from utils.metrics import ap_per_class
    precision, recall, ap, f1, _ = ap_per_class(correct, confidence, pred_cls, target_cls, plot=False, save_dir=OUT, names={0: "drone"})
    return {"precision": float(precision.mean()), "recall": float(recall.mean()), "map50": float(ap[:, 0].mean()), "map5095": float(ap.mean(1).mean()), "f1": float(f1.mean())}


def flat_stats(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int, int, str, float]], int]:
    iouv = torch.linspace(0.5, 0.95, 10)
    correct_parts: list[np.ndarray] = []
    pred_classes: list[float] = []
    target_classes: list[float] = []
    locations: list[tuple[str, int, int, str, float]] = []
    labels_total = 0
    for image_id in sorted(data):
        item = data[image_id]
        detections: list[list[float]] = []
        indices: list[int] = []
        for index, row in enumerate(item.get("detections", [])):
            converted = row_to_det(row)
            if converted is not None:
                detections.append(converted)
                indices.append(index)
        labels = [converted for row in item.get("labels", []) if (converted := row_to_label(row)) is not None]
        det_tensor = torch.tensor(detections, dtype=torch.float32) if detections else torch.zeros((0, 6), dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32) if labels else torch.zeros((0, 5), dtype=torch.float32)
        correct_parts.append(process_batch(det_tensor, label_tensor, iouv).numpy())
        pred_classes.extend(det_tensor[:, 5].tolist() if det_tensor.numel() else [])
        target_classes.extend(label_tensor[:, 0].tolist() if label_tensor.numel() else [])
        labels_total += len(labels)
        sequence, frame_id, _ = image_key(str(image_id), 0)
        for index, detection in zip(indices, detections):
            locations.append((sequence, frame_id, index, str(image_id), float(detection[4])))
    return np.concatenate(correct_parts), np.asarray(pred_classes, dtype=np.float32), np.asarray(target_classes, dtype=np.float32), locations, labels_total


def load_final_scores(split: str, data: dict[str, Any], locations: list[tuple[str, int, int, str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v53_summary = json.loads(V53_SUMMARY.read_text(encoding="utf-8"))["validation_selection"]
    base_path = V46 / ("val_oof_scores.jsonl" if split == "val" else "test_scores.jsonl")
    expert_path = V52 / ("val_expert_scores.jsonl" if split == "val" else "test_expert_scores.jsonl")
    base, _ = load_row_scores(base_path, "action_chunk_neighbor_score", 1)
    expert, _ = load_row_scores(expert_path, "action_chunk_multi_expert_score", 1)
    gates = temporal_gate_map(data, float(v53_summary["threshold"]), float(v53_summary["window_seconds"]), float(v53_summary["min_fraction"]), json.loads(FPS.read_text(encoding="utf-8")))

    v119_summary = json.loads((V119 / "official_summary.json").read_text(encoding="utf-8"))
    selected_ratio = float(v119_summary["selected_ratio"])
    selected_record = next(record for record in v119_summary["ratios"] if abs(float(record["ratio"]) - selected_ratio) < 1e-9)
    v119_config = v119_summary["validation_selection"]
    v119_path = Path(selected_record["val_scores"] if split == "val" else selected_record["test_scores"])
    v119_map, _ = load_row_scores(v119_path, str(selected_record["field"]), 1)

    scores53: list[float] = []
    scores119: list[float] = []
    for sequence, frame_id, index, image_id, raw in locations:
        key = (sequence, frame_id, index)
        base_score = max(1e-9, float(base.get(key, raw)))
        expert_score = max(1e-9, float(expert.get(key, base_score)))
        if gates.get(image_id, False):
            weight = float(v53_summary["expert_weight"])
            auxiliary = math.exp((1.0 - weight) * math.log(base_score) + weight * math.log(expert_score))
        else:
            auxiliary = base_score
        scores53.append(fuse_score(raw, auxiliary, float(v53_summary["alpha"]), "geom-mix"))
        scores119.append(fuse_score(raw, float(v119_map.get(key, raw)), float(v119_config["alpha"]), str(v119_config["mode"])))
    return np.asarray(scores53), np.asarray(scores119)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report("wait_models", 0)
    wait_for(V119 / "official_summary.json", UPSTREAM_RUN)

    report("select_on_validation", 1)
    val_data = load_predictionsgt(VAL)
    correct, pred_cls, target_cls, locations, labels_total = flat_stats(val_data)
    val_scores = list(load_final_scores("val", val_data, locations))
    rows: list[dict[str, Any]] = []
    grid = [value / 20.0 for value in range(21)]
    for mode in ("logit", "geom", "linear"):
        for weight119 in grid:
            weights = (1.0 - weight119, weight119)
            result = {"mode": mode, "weights": {"v53": weights[0], "v119": weights[1]}, **metrics(correct, blend(val_scores, weights, mode), pred_cls, target_cls, Path(r"D:\urap_modal_stage\TransVisDrone"))}
            rows.append(result)
    best = max(rows, key=lambda row: float(row["map50"]))
    (OUT / "val_sweep.json").write_text(json.dumps({"best": best, "top": sorted(rows, key=lambda row: -float(row["map50"]))[:30], "rows": rows, "labels": labels_total}, indent=2), encoding="utf-8")

    report("fixed_test", 2, validation_selection=best)
    test_data = load_predictionsgt(TEST)
    test_correct, test_pred_cls, test_target_cls, test_locations, test_labels = flat_stats(test_data)
    test_scores = list(load_final_scores("test", test_data, test_locations))
    weights = (float(best["weights"]["v53"]), float(best["weights"]["v119"]))
    test_metrics = metrics(test_correct, blend(test_scores, weights, str(best["mode"])), test_pred_cls, test_target_cls, Path(r"D:\urap_modal_stage\TransVisDrone"))
    gain = 100.0 * (test_metrics["map50"] - VATD_MAP50)
    summary = {"protocol": "OOF validation-selected confidence stack; fixed test", "validation_selection": best, "test_fixed": {**test_metrics, "labels": test_labels, "detections": len(test_locations)}, "vatd_map50": VATD_MAP50, "gain_over_vatd_points": gain, "target_3_to_5_met": 3.0 <= gain <= 5.0}
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 3, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





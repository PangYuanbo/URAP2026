from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(r"C:\Users\aaron\Desktop\URAP")
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from tools.run_tvd_oof_stack_v130 import VAL, TEST, blend, flat_stats, load_final_scores, load_predictionsgt, metrics
from tools.sweep_tvd_predictionsgt_score_fusion import fuse_score, load_row_scores

RUN = ROOT / "artifacts" / "detached_tvd_v130_v135_stack_v136"
OUT = Path(r"D:\URAP_vatd_rank_results\tvd_v130_v135_stack_v136")
V130 = Path(r"D:\URAP_vatd_rank_results\tvd_oof_stack_v130")
V135 = Path(r"D:\URAP_vatd_rank_results\tvd_train_only_action_v135")
TVD_ROOT = Path(r"D:\urap_modal_stage\TransVisDrone")
VATD_MAP50 = 0.93844


def report(stage: str, done: int, total: int = 3, **extra: object) -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    payload = {"stage": stage, "done": done, "total": total, "updated": datetime.now().astimezone().isoformat(), **extra}
    (RUN / "progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload), flush=True)


def route_scores(split: str, data: dict[str, Any], locations: list[tuple[str, int, int, str, float]]) -> tuple[np.ndarray, np.ndarray]:
    v130 = json.loads((V130 / "official_summary.json").read_text(encoding="utf-8"))["validation_selection"]
    component_scores = list(load_final_scores(split, data, locations))
    component_weights = (float(v130["weights"]["v53"]), float(v130["weights"]["v126"]), float(v130["weights"]["v129"]))
    best_stack = blend(component_scores, component_weights, str(v130["mode"]))
    v135 = json.loads((V135 / "official_summary.json").read_text(encoding="utf-8"))["validation_selection"]
    score_path = V135 / ("val_scores.jsonl" if split == "val" else "test_scores.jsonl")
    score_map, _ = load_row_scores(score_path, "tvd_train_only_action_score", 1)
    train_only = np.asarray([
        fuse_score(raw, float(score_map.get((sequence, frame_id, index), raw)), float(v135["alpha"]), str(v135["mode"]))
        for sequence, frame_id, index, _image_id, raw in locations
    ], dtype=np.float64)
    return best_stack, train_only


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report("select_validation", 0)
    val_data = load_predictionsgt(VAL)
    correct, pred_cls, target_cls, locations, labels = flat_stats(val_data)
    routes = route_scores("val", val_data, locations)
    rows: list[dict[str, Any]] = []
    for mode in ("logit", "geom", "linear"):
        for step in range(101):
            weight135 = step / 100.0
            weights = (1.0 - weight135, weight135)
            result = metrics(correct, blend(list(routes), weights, mode), pred_cls, target_cls, TVD_ROOT)
            rows.append({"mode": mode, "weights": {"v130": weights[0], "v135": weights[1]}, **result})
    best = max(rows, key=lambda row: float(row["map50"]))
    (OUT / "val_sweep.json").write_text(json.dumps({"best": best, "top": sorted(rows, key=lambda row: -float(row["map50"]))[:30], "labels": labels}, indent=2), encoding="utf-8")
    report("fixed_test", 2, validation_selection=best)
    test_data = load_predictionsgt(TEST)
    test_correct, test_pred_cls, test_target_cls, test_locations, test_labels = flat_stats(test_data)
    test_routes = route_scores("test", test_data, test_locations)
    weights = (float(best["weights"]["v130"]), float(best["weights"]["v135"]))
    test_result = metrics(test_correct, blend(list(test_routes), weights, str(best["mode"])), test_pred_cls, test_target_cls, TVD_ROOT)
    gain = 100.0 * (test_result["map50"] - VATD_MAP50)
    summary = {"protocol": "validation-selected blend of V130 OOF stack and train-only V135; fixed test", "validation_selection": best, "test_fixed": {**test_result, "labels": test_labels, "detections": len(test_locations)}, "vatd_map50": VATD_MAP50, "gain_over_vatd_points": gain, "target_3_to_5_met": 3.0 <= gain <= 5.0}
    (OUT / "official_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report("done", 3, summary=summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import copy
import json
import pickle
from pathlib import Path
from typing import Any

from eval_tvd_predictionsgt_pkl import load_predictionsgt, main as _unused_eval_main
from rescore_li_tetc_diagnostics_from_tracklets import adjusted_score, load_tracklet_scores


def parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in text.replace(",", " ").split() if part.strip()]


def image_key(image_id: str, pred_index: int) -> tuple[str, int, int]:
    parts = str(image_id).split("_")
    if len(parts) >= 3 and parts[0] == "Clip":
        seq = f"{parts[0]}_{parts[1]}"
        try:
            frame_id = int(parts[2])
        except ValueError:
            frame_id = 0
        return seq, frame_id, int(pred_index)
    return str(image_id), 0, int(pred_index)


def clone_with_scores(
    data: dict[str, Any],
    score_map: dict[tuple[str, int, int], float],
    center: float,
    beta: float,
    mode: str,
    missing_score_behavior: str,
    score_gate: float | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for image_id, item in data.items():
        new_item = {"labels": item.get("labels", []), "detections": []}
        for pred_index, row in enumerate(item.get("detections", [])):
            action_score = score_map.get(image_key(str(image_id), pred_index))
            if action_score is None:
                if missing_score_behavior == "drop":
                    continue
                new_item["detections"].append(row)
                continue
            new_row = dict(row)
            raw_score = float(new_row.get("score", 0.0))
            if mode == "gated-boost-low":
                if score_gate is None or score_gate <= 0:
                    raise ValueError("gated-boost-low requires a positive score_gate")
                gate = max(0.0, 1.0 - raw_score / float(score_gate))
                new_score = raw_score + float(beta) * max(0.0, float(action_score) - float(center)) * gate
                new_row["score"] = float(min(1.0, max(0.0, new_score)))
                new_row["score_gate"] = float(score_gate)
            else:
                new_row["score"] = float(adjusted_score(raw_score, float(action_score), center, beta, mode, 0.0, 1.0))
            new_row["raw_score"] = raw_score
            new_row["vatd_score"] = float(action_score)
            new_item["detections"].append(new_row)
        out[image_id] = new_item
    return out


def evaluate_data(data: dict[str, Any], tvd_root: Path, out_dir: Path) -> dict[str, Any]:
    import numpy as np
    import sys
    import torch

    sys.path.insert(0, str(tvd_root.resolve()))
    if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]
    from eval_tvd_predictionsgt_pkl import process_batch, row_to_det, row_to_label
    from utils.metrics import ap_per_class  # type: ignore

    iouv = torch.linspace(0.5, 0.95, 10)
    stats = []
    images = labels_total = detections_total = 0
    for image_id in sorted(data):
        images += 1
        item = data[image_id]
        det_rows = [row_to_det(row) for row in item.get("detections", [])]
        label_rows = [row_to_label(row) for row in item.get("labels", [])]
        det_rows = [row for row in det_rows if row is not None]
        label_rows = [row for row in label_rows if row is not None]
        detections_total += len(det_rows)
        labels_total += len(label_rows)
        det = torch.tensor(det_rows, dtype=torch.float32) if det_rows else torch.zeros((0, 6), dtype=torch.float32)
        labels = torch.tensor(label_rows, dtype=torch.float32) if label_rows else torch.zeros((0, 5), dtype=torch.float32)
        correct = process_batch(det, labels, iouv)
        tcls = labels[:, 0].tolist() if labels.numel() else []
        stats.append((correct.numpy(), det[:, 4].numpy(), det[:, 5].numpy(), np.asarray(tcls)))
    arrays = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(arrays) and arrays[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*arrays, plot=False, save_dir=out_dir, names={0: "drone"})
        return {
            "images": images,
            "labels": labels_total,
            "detections": detections_total,
            "precision": float(p.mean()),
            "recall": float(r.mean()),
            "map50": float(ap[:, 0].mean()),
            "map5095": float(ap.mean(1).mean()),
            "f1": float(f1.mean()),
        }
    return {"images": images, "labels": labels_total, "detections": detections_total, "precision": 0.0, "recall": 0.0, "map50": 0.0, "map5095": 0.0, "f1": 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep action-score rescoring for TransVisDrone predictionsgt pkl.")
    parser.add_argument("--tvd-root", type=Path, default=Path("papers/TransVisDrone"))
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--tracklet-jsonl", type=Path, required=True)
    parser.add_argument("--score-field", default="vatd_score")
    parser.add_argument("--centers", default="0.01 0.03 0.05 0.07 0.10 0.15 0.20")
    parser.add_argument("--betas", default="0.02 0.05 0.10 0.20 0.30 0.40")
    parser.add_argument("--modes", nargs="*", default=["additive", "boost-only", "suppress-only"])
    parser.add_argument("--score-gates", default="0.05")
    parser.add_argument("--missing-score-behaviors", nargs="*", default=["keep"])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--write-best-pkl", type=Path, default=None)
    args = parser.parse_args()

    data = load_predictionsgt(args.predictionsgt_pkl.resolve())
    score_map, score_summary = load_tracklet_scores(args.tracklet_jsonl.resolve(), args.score_field, min_tracklet_rows=1)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    out_dir = args.out_json.parent
    for missing in args.missing_score_behaviors:
        for mode in args.modes:
            for center in parse_csv_floats(args.centers):
                for beta in parse_csv_floats(args.betas):
                    score_gates = parse_csv_floats(args.score_gates) if mode == "gated-boost-low" else [None]
                    for score_gate in score_gates:
                        rescored = clone_with_scores(data, score_map, center, beta, mode, missing, score_gate)
                        metrics = evaluate_data(rescored, args.tvd_root, out_dir)
                        row = {
                            "mode": mode,
                            "center": center,
                            "beta": beta,
                            "score_gate": score_gate,
                            "missing_score_behavior": missing,
                            **metrics,
                        }
                        rows.append(row)
                        if best is None or float(row["map50"]) > float(best["map50"]):
                            best = row
    summary = {
        "predictionsgt_pkl": str(args.predictionsgt_pkl.resolve()),
        "tracklet_jsonl": str(args.tracklet_jsonl.resolve()),
        "score_field": args.score_field,
        "score_summary": score_summary,
        "best": best,
        "top": sorted(rows, key=lambda row: (-float(row["map50"]), -float(row["recall"])))[:20],
        "rows": rows,
    }
    if args.write_best_pkl is not None and best is not None:
        best_data = clone_with_scores(
            data,
            score_map,
            float(best["center"]),
            float(best["beta"]),
            str(best["mode"]),
            str(best["missing_score_behavior"]),
            best.get("score_gate"),
        )
        args.write_best_pkl.parent.mkdir(parents=True, exist_ok=True)
        with args.write_best_pkl.open("wb") as f:
            pickle.dump(best_data, f)
        summary["best_pkl"] = str(args.write_best_pkl.resolve())
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["best", "top"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

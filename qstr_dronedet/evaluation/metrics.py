from __future__ import annotations

import json
from pathlib import Path

from qstr_dronedet.candidates.merge import bbox_iou


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def evaluate_predictions(pred_path: str | Path, gt_path: str | Path | None = None, out: str | Path | None = None) -> dict:
    preds = _load_jsonl(pred_path)
    metrics = {"num_predictions": len(preds), "num_drone_predictions": sum(1 for p in preds if p.get("predicted_class") == "drone")}
    if gt_path and Path(gt_path).exists():
        gt = json.loads(Path(gt_path).read_text(encoding="utf-8"))
        gt_items = gt.get("annotations", gt if isinstance(gt, list) else [])
        matched03 = matched05 = 0
        for ann in gt_items:
            frame_id = ann.get("frame_id", ann.get("image_id", 0))
            box = ann.get("bbox_xyxy")
            if box is None and "bbox" in ann:
                x, y, w, h = ann["bbox"]
                box = [x, y, x + w, y + h]
            frame_preds = [p for p in preds if p.get("frame_id") == frame_id]
            best = max([bbox_iou(tuple(p["bbox"]), tuple(box)) for p in frame_preds], default=0.0)
            matched03 += int(best >= 0.3)
            matched05 += int(best >= 0.5)
        n = max(1, len(gt_items))
        metrics.update({"candidate_recall_iou03": matched03 / n, "candidate_recall_iou05": matched05 / n})
    if out:
        outp = Path(out)
        outp.mkdir(parents=True, exist_ok=True)
        (outp / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


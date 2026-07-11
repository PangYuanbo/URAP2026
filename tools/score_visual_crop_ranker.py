from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet34

from eval_tvd_predictionsgt_pkl import load_predictionsgt, row_to_det
from sweep_tvd_predictionsgt_action_rescore import image_key
from train_visual_crop_ranker import crop_context


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictionsgt-pkl", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--input-tracklets", type=Path, required=True)
    parser.add_argument("--output-tracklets", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--score-field", default="visual_crop_score")
    parser.add_argument("--min-score", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    image_size = int(checkpoint["image_size"])
    context_scale = float(checkpoint["context_scale"])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    data = load_predictionsgt(args.predictionsgt_pkl)
    score_map: dict[tuple[str, int, int], float] = {}
    batch_images: list[torch.Tensor] = []
    batch_keys: list[tuple[str, int, int]] = []
    scored = 0

    def flush() -> None:
        nonlocal scored
        if not batch_images:
            return
        tensor = torch.stack(batch_images).to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            values = torch.sigmoid(model(tensor).squeeze(1)).float().cpu().numpy()
        for key, value in zip(batch_keys, values.tolist(), strict=True):
            score_map[key] = float(value)
        scored += len(batch_images)
        batch_images.clear()
        batch_keys.clear()

    total_images = len(data)
    for image_number, (image_id, item) in enumerate(data.items(), start=1):
        frame_path = args.frame_root / f"{image_id}.png"
        if not frame_path.is_file():
            raise FileNotFoundError(frame_path)
        with Image.open(frame_path) as source:
            image = source.convert("RGB")
            for pred_index, row in enumerate(item.get("detections", [])):
                detection = row_to_det(row)
                if detection is None or float(detection[4]) < args.min_score:
                    continue
                batch_images.append(transform(crop_context(image, detection[:4], context_scale)))
                batch_keys.append(image_key(str(image_id), pred_index))
                if len(batch_images) >= args.batch_size:
                    flush()
        if args.progress_json and (image_number % 100 == 0 or image_number == total_images):
            args.progress_json.parent.mkdir(parents=True, exist_ok=True)
            args.progress_json.write_text(json.dumps({"stage": "score", "done": image_number, "total": total_images, "scored": scored}), encoding="utf-8")
            print(json.dumps({"kind": "visual_crop_score_progress", "done": image_number, "total": total_images, "scored": scored, "device": str(device)}), flush=True)
    flush()

    args.output_tracklets.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    matched = 0
    with args.input_tracklets.open("r", encoding="utf-8-sig") as source, args.output_tracklets.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            for row in item.get("rows") or []:
                key = (str(row.get("seq")), int(float(row.get("frame_id", 0))), int(float(row.get("prediction_index", 0))))
                value = score_map.get(key)
                row[args.score_field] = float(value) if value is not None else 0.0
                matched += int(value is not None)
                rows += 1
            target.write(json.dumps(item, separators=(",", ":")) + "\n")
    summary = {"device": str(device), "images": total_images, "detections_scored": scored, "rows": rows, "rows_matched": matched, "score_field": args.score_field, "output": str(args.output_tracklets)}
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

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
    parser.add_argument("--out-score-map", type=Path, required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    parser.add_argument("--download-pid", type=int, required=True)
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
    if args.out_score_map.is_file():
        with args.out_score_map.open("rb") as handle:
            score_map = pickle.load(handle)
    completed_images: set[str] = set()
    for seq, frame_id, _pred_index in score_map:
        completed_images.add(f"{seq}_{frame_id:05d}")

    def process_image(image_id: str, item: dict[str, object]) -> int:
        frame_path = args.frame_root / f"{image_id}.png"
        if not frame_path.is_file():
            return 0
        tensors: list[torch.Tensor] = []
        keys: list[tuple[str, int, int]] = []
        try:
            with Image.open(frame_path) as source:
                image = source.convert("RGB")
                for pred_index, row in enumerate(item.get("detections", [])):
                    detection = row_to_det(row)
                    if detection is None or float(detection[4]) < args.min_score:
                        continue
                    tensors.append(transform(crop_context(image, detection[:4], context_scale)))
                    keys.append(image_key(image_id, pred_index))
        except (OSError, PermissionError, SyntaxError):
            return 0
        for start in range(0, len(tensors), args.batch_size):
            batch = torch.stack(tensors[start : start + args.batch_size]).to(device, non_blocking=True)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                values = torch.sigmoid(model(batch).squeeze(1)).float().cpu().numpy()
            for key, value in zip(keys[start : start + args.batch_size], values.tolist(), strict=True):
                score_map[key] = float(value)
        completed_images.add(image_id)
        return len(keys)

    args.out_score_map.parent.mkdir(parents=True, exist_ok=True)
    idle_rounds = 0
    while len(completed_images) < len(data):
        new_images = 0
        new_scores = 0
        for image_id, item in data.items():
            if image_id in completed_images:
                continue
            frame_path = args.frame_root / f"{image_id}.png"
            if not frame_path.is_file():
                continue
            new_scores += process_image(str(image_id), item)
            new_images += 1
            if new_images % 50 == 0:
                with args.out_score_map.open("wb") as handle:
                    pickle.dump(score_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
                progress = {"stage": "score_stream", "done": len(completed_images), "total": len(data), "scores": len(score_map), "new_images": new_images, "device": str(device)}
                args.progress_json.write_text(json.dumps(progress), encoding="utf-8")
                print(json.dumps(progress), flush=True)
        with args.out_score_map.open("wb") as handle:
            pickle.dump(score_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
        progress = {"stage": "score_stream", "done": len(completed_images), "total": len(data), "scores": len(score_map), "new_images": new_images, "new_scores": new_scores, "device": str(device)}
        args.progress_json.write_text(json.dumps(progress), encoding="utf-8")
        print(json.dumps(progress), flush=True)
        if new_images:
            idle_rounds = 0
        else:
            idle_rounds += 1
        if idle_rounds >= 12:
            try:
                import psutil

                download_running = psutil.pid_exists(args.download_pid)
            except ImportError:
                download_running = True
            if not download_running:
                raise RuntimeError(f"download stopped with only {len(completed_images)}/{len(data)} images scored")
        time.sleep(10)
    print(json.dumps({"stage": "done", "done": len(completed_images), "total": len(data), "scores": len(score_map), "device": str(device)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

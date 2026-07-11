from __future__ import annotations

import json
import pickle
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(r"D:\URAP_vatd_rank_results\nps_visual_crop_v1\model.pt")
PREDICTIONS = Path(r"D:\URAP_vatd_rank_inputs\nps_predictionsgt_split_0.pkl")
app = modal.App("urap-nps-visual-crop-score-v1")
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install("torchvision==0.20.1", "pillow==11.0.0", "numpy==1.26.4")
    .add_local_file(MODEL, "/workspace/model.pt", copy=True)
    .add_local_file(PREDICTIONS, "/workspace/predictions.pkl", copy=True)
    .add_local_file(ROOT / "tools/eval_tvd_predictionsgt_pkl.py", "/workspace/eval_tvd_predictionsgt_pkl.py", copy=True)
    .add_local_file(ROOT / "tools/train_visual_crop_ranker.py", "/workspace/train_visual_crop_ranker.py", copy=True)
)
source = modal.Volume.from_name("urap-nps-formatted-v1")
results = modal.Volume.from_name("urap-nps-visual-crop-results-v1", create_if_missing=True)


@app.function(image=image, gpu="L40S", cpu=16, memory=65536, volumes={"/source": source, "/results": results}, timeout=6 * 60 * 60)
def score() -> dict[str, object]:
    import sys

    import torch
    from PIL import Image
    from torchvision import transforms
    from torchvision.models import resnet34

    sys.path.insert(0, "/workspace")
    from eval_tvd_predictionsgt_pkl import load_predictionsgt, row_to_det
    from train_visual_crop_ranker import crop_context

    source.reload()
    results.reload()
    checkpoint = torch.load("/workspace/model.pt", map_location="cpu", weights_only=True)
    model = resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda")
    model.to(device).eval()
    image_size = int(checkpoint["image_size"])
    context_scale = float(checkpoint["context_scale"])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    data = load_predictionsgt(Path("/workspace/predictions.pkl"))
    output: dict[tuple[str, int, int], float] = {}
    batch_images: list[torch.Tensor] = []
    batch_keys: list[tuple[str, int, int]] = []
    scored = 0

    def key_for(image_id: str, pred_index: int) -> tuple[str, int, int]:
        parts = image_id.split("_")
        return f"{parts[0]}_{parts[1]}", int(parts[2]), pred_index

    def flush() -> None:
        nonlocal scored
        if not batch_images:
            return
        tensor = torch.stack(batch_images).to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            values = torch.sigmoid(model(tensor).squeeze(1)).float().cpu().numpy()
        output.update((key, float(value)) for key, value in zip(batch_keys, values.tolist(), strict=True))
        scored += len(batch_images)
        batch_images.clear()
        batch_keys.clear()

    total = len(data)
    for image_number, (image_id, item) in enumerate(data.items(), start=1):
        frame = Path("/source/NPS/AllFrames/test") / f"{image_id}.png"
        with Image.open(frame) as source_image:
            rgb = source_image.convert("RGB")
            for pred_index, row in enumerate(item.get("detections", [])):
                detection = row_to_det(row)
                if detection is None or float(detection[4]) < 0.005:
                    continue
                batch_images.append(transform(crop_context(rgb, detection[:4], context_scale)))
                batch_keys.append(key_for(str(image_id), pred_index))
                if len(batch_images) >= 1024:
                    flush()
        if image_number % 250 == 0 or image_number == total:
            progress = {"stage": "score", "done": image_number, "total": total, "scored": scored, "device": str(device)}
            Path("/results/progress.json").write_text(json.dumps(progress), encoding="utf-8")
            results.commit()
            print(json.dumps(progress), flush=True)
    flush()
    output_path = Path("/results/visual_score_map.pkl")
    with output_path.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    summary = {"stage": "done", "done": total, "total": total, "scored": scored, "score_map": str(output_path), "score_map_bytes": output_path.stat().st_size}
    Path("/results/summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    results.commit()
    print(json.dumps(summary), flush=True)
    return summary


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(score.remote(), indent=2))

#!/usr/bin/env python3
"""Evaluate a frozen-feature bbox readout and its mask-to-box reference."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qstr_dronedet.samurai_bbox_readout import BBoxReadout, decode_delta, normalized_previous_box, tracking_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def predict_boxes(
    model: BBoxReadout,
    normalized_pointer: np.ndarray,
    target: np.ndarray,
    previous_source: np.ndarray | None,
    image_wh: np.ndarray,
    sequence_id: np.ndarray,
    frame_index: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Run the bbox head autoregressively or with a supplied previous-box source."""
    predictions = np.zeros_like(target)
    with torch.inference_mode():
        for sequence in np.unique(sequence_id):
            indices = np.flatnonzero(sequence_id == sequence)
            indices = indices[np.argsort(frame_index[indices])]
            previous = target[indices[0]].copy()
            predictions[indices[0]] = previous
            for index in indices[1:]:
                if previous_source is not None:
                    previous = previous_source[index]
                previous_feature = normalized_previous_box(previous[None], image_wh[index : index + 1])
                predicted_delta = model(
                    torch.from_numpy(normalized_pointer[index : index + 1]).to(device),
                    torch.from_numpy(previous_feature).to(device),
                ).cpu().numpy()[0]
                prediction = decode_delta(previous[None], predicted_delta[None], image_wh[index : index + 1])[0]
                predictions[index] = prediction
                if previous_source is None:
                    previous = prediction
    return predictions


def previous_ground_truth_boxes(
    target: np.ndarray, sequence_id: np.ndarray, frame_index: np.ndarray
) -> np.ndarray:
    previous = target.copy()
    for sequence in np.unique(sequence_id):
        indices = np.flatnonzero(sequence_id == sequence)
        indices = indices[np.argsort(frame_index[indices])]
        if len(indices) > 1:
            previous[indices[1:]] = target[indices[:-1]]
    return previous


def main() -> int:
    args = parse_args()
    data = np.load(args.features)
    pointer = np.asarray(data["object_pointer"], dtype=np.float32)
    target = np.asarray(data["target_xywh"], dtype=np.float32)
    mask_boxes = np.asarray(data["mask_xywh"], dtype=np.float32)
    image_wh = np.asarray(data["image_wh"], dtype=np.float32)
    sequence_id = np.asarray(data["sequence_id"])
    frame_index = np.asarray(data["frame_index"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = BBoxReadout(pointer_dim=int(checkpoint["pointer_dim"])).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    pointer_mean = checkpoint["pointer_mean"].numpy()
    pointer_std = checkpoint["pointer_std"].numpy()
    normalized_pointer = (pointer - pointer_mean) / pointer_std

    mask_previous = np.asarray(data["previous_xywh"], dtype=np.float32)
    gt_previous = previous_ground_truth_boxes(target, sequence_id, frame_index)
    predictions = predict_boxes(
        model, normalized_pointer, target, None, image_wh, sequence_id, frame_index, device
    )
    mask_conditioned_predictions = predict_boxes(
        model, normalized_pointer, target, mask_previous, image_wh, sequence_id, frame_index, device
    )
    gt_conditioned_predictions = predict_boxes(
        model, normalized_pointer, target, gt_previous, image_wh, sequence_id, frame_index, device
    )
    mask_reference = mask_boxes.copy()
    for sequence in np.unique(sequence_id):
        indices = np.flatnonzero(sequence_id == sequence)
        first = indices[np.argmin(frame_index[indices])]
        mask_reference[first] = target[first]

    report = {
        "bbox_readout": asdict(tracking_metrics(predictions, target)),
        "bbox_readout_mask_conditioned": asdict(tracking_metrics(mask_conditioned_predictions, target)),
        "bbox_readout_gt_conditioned": asdict(tracking_metrics(gt_conditioned_predictions, target)),
        "mask_to_box_reference": asdict(tracking_metrics(mask_reference, target)),
        "frames": len(target),
        "sequences": int(len(np.unique(sequence_id))),
        "feature_source": str(args.features.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "sequence_results": [],
    }
    for sequence in np.unique(sequence_id):
        indices = np.flatnonzero(sequence_id == sequence)
        report["sequence_results"].append(
            {
                "sequence_id": int(sequence),
                "bbox_readout": asdict(tracking_metrics(predictions[indices], target[indices])),
                "bbox_readout_mask_conditioned": asdict(
                    tracking_metrics(mask_conditioned_predictions[indices], target[indices])
                ),
                "bbox_readout_gt_conditioned": asdict(
                    tracking_metrics(gt_conditioned_predictions[indices], target[indices])
                ),
                "mask_to_box_reference": asdict(tracking_metrics(mask_reference[indices], target[indices])),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        args.output.with_suffix(".predictions.npz"),
        prediction_xywh=predictions,
        mask_conditioned_prediction_xywh=mask_conditioned_predictions,
        gt_conditioned_prediction_xywh=gt_conditioned_predictions,
        mask_reference_xywh=mask_reference,
        target_xywh=target,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

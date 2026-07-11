from __future__ import annotations

import csv
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from qstr_dronedet.native_video_detector import NPSClipDataset
from tools.export_native_video_predictionsgt import (
    append_action_chunk_detections,
    build_export_frame_maps,
    merge_action_chunk_support,
    nms_detections,
    samurai_motion_rerank,
    samurai_tracklet_rerank,
)


def _write_synthetic_nps(root: Path, frames: int = 8) -> tuple[Path, Path]:
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True)
    gt_csv = root / "gt.csv"
    for frame_id in range(1, frames + 1):
        img = Image.new("RGB", (96, 64), color=(30, frame_id * 20 % 255, 50))
        img.save(frames_dir / f"Clip_1_{frame_id:05d}.png")
    with gt_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq", "frame_id", "x1", "y1", "x2", "y2", "video_path"])
        writer.writeheader()
        for frame_id in range(1, frames + 1):
            writer.writerow(
                {
                    "seq": "Clip_1",
                    "frame_id": frame_id,
                    "x1": 10 + frame_id,
                    "y1": 12,
                    "x2": 24 + frame_id,
                    "y2": 26,
                    "video_path": f"Clip_1/Clip_1_{frame_id:05d}.png",
                }
            )
    return frames_dir, gt_csv


def test_native_video_export_predictionsgt_roundtrip(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    frames_dir, gt_csv = _write_synthetic_nps(tmp_path / "data")
    out_dir = tmp_path / "run"
    weights = out_dir / "native_video_detector.pt"
    pkl_path = tmp_path / "predictionsgt.pkl"
    eval_json = tmp_path / "eval.json"
    sweep_json = tmp_path / "threshold_sweep.json"
    sweep_csv = tmp_path / "threshold_sweep.csv"

    train_cmd = [
        sys.executable,
        str(repo / "tools" / "train_native_video_detector.py"),
        "--frames-dir",
        str(frames_dir),
        "--gt-csv",
        str(gt_csv),
        "--out-dir",
        str(out_dir),
        "--max-samples",
        "4",
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--image-size",
        "64",
        "--d-model",
        "32",
        "--encoder-layers",
        "1",
        "--decoder-layers",
        "1",
        "--num-queries",
        "4",
        "--query-mode",
        "dense",
        "--dense-obj-source",
        "conv",
        "--quality-score-mode",
        "iou",
        "--quality-loss-weight",
        "0.1",
        "--quality-hard-negative-topk",
        "8",
        "--quality-focal-gamma",
        "0.0",
        "--dense-positive-topk",
        "1",
        "--future-len",
        "2",
        "--clip-len",
        "4",
        "--num-workers",
        "0",
        "--ema",
        "--device",
        "cpu",
    ]
    subprocess.run(train_cmd, cwd=repo, check=True)
    assert (out_dir / "summary.json").exists()
    assert not list(out_dir.glob("*.tmp.*"))

    export_cmd = [
        sys.executable,
        str(repo / "tools" / "export_native_video_predictionsgt.py"),
        "--weights",
        str(weights),
        "--frames-dir",
        str(frames_dir),
        "--gt-csv",
        str(gt_csv),
        "--out-pkl",
        str(pkl_path),
        "--batch-size",
        "2",
        "--top-k",
        "4",
        "--quality-score-weight",
        "0.5",
        "--device",
        "cpu",
    ]
    subprocess.run(export_cmd, cwd=repo, check=True)
    export_summary = json.loads(pkl_path.with_suffix(".summary.json").read_text(encoding="utf-8"))
    assert export_summary["state_key"] == "ema_model"
    assert export_summary["max_samples"] == 0
    assert export_summary["full_split"] is True
    assert export_summary["memory_attention"] == "none"
    assert export_summary["memory_slots"] == 64
    assert export_summary["memory_match_mode"] == "none"
    assert export_summary["memory_match_weight"] == 0.0
    assert export_summary["memory_match_temperature"] == 5.0
    assert export_summary["motion_score_mode"] == "none"
    assert export_summary["motion_score_weight"] == 1.0
    assert export_summary["proposal_mode"] == "none"
    assert export_summary["proposal_prefilter_topk"] == 0
    assert export_summary["proposal_score_weight"] == 0.0
    assert export_summary["quality_score_mode"] == "iou"
    assert export_summary["quality_score_weight"] == 0.5
    assert export_summary["samurai_motion_rerank"] is False
    assert export_summary["samurai_appearance_weight"] == 0.6
    assert export_summary["samurai_motion_iou_weight"] == 0.3
    assert export_summary["samurai_center_weight"] == 0.05
    assert export_summary["samurai_tracklet_rerank"] is False
    assert export_summary["samurai_tracklet_candidate_topk"] == 32
    assert export_summary["samurai_tracklet_weight"] == 0.35
    assert export_summary["action_chunk_backfill"] is False
    assert export_summary["action_chunk_backfilled_detections"] == 0
    assert export_summary["action_chunk_score_decay"] == 0.85
    assert export_summary["action_chunk_merge_mode"] == "add"
    assert export_summary["action_chunk_supported_matches"] == 0
    with pkl_path.open("rb") as f:
        predictionsgt = pickle.load(f)
    assert len(predictionsgt) == 8
    assert sum(len(item["labels"]) for item in predictionsgt.values()) == 8
    assert sum(len(item["detections"]) for item in predictionsgt.values()) > 0
    assert any("quality_score" in det for item in predictionsgt.values() for det in item["detections"])

    eval_cmd = [
        sys.executable,
        str(repo / "tools" / "eval_tvd_predictionsgt_pkl.py"),
        "--predictionsgt-pkl",
        str(pkl_path),
        "--out-json",
        str(eval_json),
    ]
    subprocess.run(eval_cmd, cwd=repo, check=True)
    assert eval_json.exists()

    sweep_cmd = [
        sys.executable,
        str(repo / "tools" / "sweep_tvd_predictionsgt_thresholds.py"),
        "--predictionsgt-pkl",
        str(pkl_path),
        "--out-json",
        str(sweep_json),
        "--out-csv",
        str(sweep_csv),
        "--score-thresholds",
        "0.0",
        "0.001",
        "--top-ks",
        "2",
        "4",
    ]
    subprocess.run(sweep_cmd, cwd=repo, check=True)
    assert sweep_json.exists()
    assert sweep_csv.exists()


def test_native_video_export_nms_suppresses_duplicate_boxes() -> None:
    detections = [
        {"bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.9, "category_id": 0},
        {"bbox": [11.0, 11.0, 21.0, 21.0], "score": 0.8, "category_id": 0},
        {"bbox": [40.0, 40.0, 50.0, 50.0], "score": 0.7, "category_id": 0},
    ]
    kept = nms_detections(detections, iou_threshold=0.5)
    assert len(kept) == 2
    assert kept[0]["score"] == 0.9
    assert kept[1]["score"] == 0.7


def test_samurai_motion_rerank_promotes_motion_consistent_candidate() -> None:
    out = {
        "Clip_1_00001": {
            "detections": [
                {"bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.90, "category_id": 0},
            ],
            "labels": [],
        },
        "Clip_1_00002": {
            "detections": [
                {"bbox": [80.0, 80.0, 90.0, 90.0], "score": 0.95, "category_id": 0},
                {"bbox": [10.5, 10.0, 20.5, 20.0], "score": 0.50, "category_id": 0},
            ],
            "labels": [],
        },
    }
    frame_order = [
        ("Clip_1_00001", "Clip_1", 1),
        ("Clip_1_00002", "Clip_1", 2),
    ]
    samurai_motion_rerank(
        out,
        frame_order,
        appearance_weight=0.2,
        motion_iou_weight=0.7,
        center_weight=0.1,
        center_sigma_pixels=32.0,
        update_score_threshold=0.05,
        update_motion_iou_threshold=0.0,
        lost_tau=8,
        velocity_momentum=0.6,
    )
    detections = out["Clip_1_00002"]["detections"]
    assert detections[0]["bbox"] == [10.5, 10.0, 20.5, 20.0]
    assert detections[0]["appearance_score"] == 0.50
    assert detections[0]["samurai_motion_iou"] > detections[1]["samurai_motion_iou"]


def test_samurai_tracklet_rerank_keeps_multiple_motion_memories() -> None:
    out = {
        "Clip_1_00001": {
            "detections": [
                {"bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.80, "category_id": 0},
                {"bbox": [50.0, 50.0, 60.0, 60.0], "score": 0.78, "category_id": 0},
            ],
            "labels": [],
        },
        "Clip_1_00002": {
            "detections": [
                {"bbox": [80.0, 80.0, 90.0, 90.0], "score": 0.95, "category_id": 0},
                {"bbox": [11.0, 10.0, 21.0, 20.0], "score": 0.45, "category_id": 0},
                {"bbox": [51.0, 50.0, 61.0, 60.0], "score": 0.43, "category_id": 0},
            ],
            "labels": [],
        },
    }
    frame_order = [
        ("Clip_1_00001", "Clip_1", 1),
        ("Clip_1_00002", "Clip_1", 2),
    ]
    samurai_tracklet_rerank(
        out,
        frame_order,
        candidate_topk=8,
        center_sigma_pixels=32.0,
        match_threshold=0.1,
        max_gap=1,
        spawn_score_threshold=0.01,
        length_norm=2.0,
        appearance_weight=0.2,
        tracklet_weight=0.8,
        unmatched_scale=0.2,
        velocity_momentum=0.6,
    )
    detections = out["Clip_1_00002"]["detections"]
    assert detections[0]["bbox"] in ([11.0, 10.0, 21.0, 20.0], [51.0, 50.0, 61.0, 60.0])
    assert detections[1]["bbox"] in ([11.0, 10.0, 21.0, 20.0], [51.0, 50.0, 61.0, 60.0])
    assert detections[0]["samurai_track_id"] != detections[1]["samurai_track_id"]
    assert detections[0]["samurai_tracklet_length"] == 2
    assert detections[1]["samurai_tracklet_length"] == 2
    assert detections[2]["bbox"] == [80.0, 80.0, 90.0, 90.0]
    assert detections[2]["samurai_tracklet_score"] == 0.0


def test_action_chunk_backfill_adds_future_frame_candidate(tmp_path: Path) -> None:
    frames_dir, gt_csv = _write_synthetic_nps(tmp_path / "data", frames=4)
    dataset = NPSClipDataset(frames_dir, gt_csv, clip_len=2, future_len=2, image_size=64, max_samples=3)
    frame_index_by_key, export_keys = build_export_frame_maps(dataset)
    out: dict[str, dict[str, list[dict[str, object]]]] = {}
    batch = {
        "seq": ["Clip_1"],
        "frame_id": [1],
        "image_id": ["Clip_1_00001"],
        "image_size": torch.tensor([[96.0, 64.0]], dtype=torch.float32),
    }
    chunk_boxes = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    chunk_logits = torch.full((1, 2, 3), -10.0, dtype=torch.float32)
    chunk_boxes[0, 0, 1] = torch.tensor([0.20, 0.30, 0.10, 0.12])
    chunk_logits[0, 0, 1] = 10.0

    inserted = append_action_chunk_detections(
        out,
        batch,
        chunk_boxes=chunk_boxes,
        chunk_logits=chunk_logits,
        dataset=dataset,
        frame_index_by_key=frame_index_by_key,
        export_keys=export_keys,
        gt=dataset.gt,
        max_step=1,
        top_k=1,
        score_threshold=0.0,
        score_decay=1.0,
    )

    assert inserted == 1
    assert "Clip_1_00002" in out
    detections = out["Clip_1_00002"]["detections"]
    assert len(detections) == 1
    assert detections[0]["source"] == "action_chunk"
    assert detections[0]["action_chunk_step"] == 1
    assert detections[0]["action_chunk_source_image_id"] == "Clip_1_00001"
    assert len(out["Clip_1_00002"]["labels"]) == 1


def test_action_chunk_support_boosts_existing_candidate_and_drops_unmatched() -> None:
    out = {
        "Clip_1_00002": {
            "detections": [
                {"bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.40, "category_id": 0},
                {
                    "bbox": [10.5, 10.0, 20.5, 20.0],
                    "score": 0.80,
                    "category_id": 0,
                    "source": "action_chunk",
                    "action_chunk_step": 1,
                },
                {
                    "bbox": [80.0, 80.0, 90.0, 90.0],
                    "score": 0.90,
                    "category_id": 0,
                    "source": "action_chunk",
                    "action_chunk_step": 1,
                },
            ],
            "labels": [],
        }
    }
    supported = merge_action_chunk_support(out, support_iou=0.3, support_weight=0.5, keep_unmatched=False)
    detections = out["Clip_1_00002"]["detections"]
    assert supported == 1
    assert len(detections) == 1
    assert detections[0].get("source") != "action_chunk"
    assert detections[0]["pre_action_chunk_score"] == 0.40
    assert abs(float(detections[0]["score"]) - 0.60) < 1e-6
    assert detections[0]["action_chunk_support_count"] == 1


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="native_video_export_") as tmp:
        test_native_video_export_predictionsgt_roundtrip(Path(tmp))
    with tempfile.TemporaryDirectory(prefix="native_video_action_chunk_") as tmp:
        test_action_chunk_backfill_adds_future_frame_candidate(Path(tmp))
    test_action_chunk_support_boosts_existing_candidate_and_drops_unmatched()
    test_native_video_export_nms_suppresses_duplicate_boxes()
    test_samurai_motion_rerank_promotes_motion_consistent_candidate()
    test_samurai_tracklet_rerank_keeps_multiple_motion_memories()

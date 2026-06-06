import json

import cv2
import numpy as np
import torch

from qstr_dronedet.tracking.video_action_policy import (
    VideoActionTrackletDataset,
    score_tracklets_with_vatd_motion_action_policy,
    score_tracklets_with_video_action_multihead_policy,
    score_tracklets_with_video_action_policy,
    train_vatd_motion_action_policy,
    train_video_action_chunk_policy,
    train_video_action_multihead_policy,
)


def _write_video_action_smoke_data(tmp_path):
    frame_root = tmp_path / "frames"
    frame_root.mkdir()
    rows = []
    for frame_id in range(6):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        x1 = 10 + frame_id
        y1 = 20
        cv2.rectangle(image, (x1, y1), (x1 + 5, y1 + 5), (255, 255, 255), -1)
        cv2.imwrite(str(frame_root / f"Clip_001_{frame_id:05d}.png"), image)
        rows.append(
            {
                "seq": "Clip_001",
                "track_id": "t1",
                "frame_id": frame_id,
                "bbox": [x1, y1, x1 + 5, y1 + 5],
                "objectness": 0.8,
                "visible": True,
            }
        )
    tracklets = tmp_path / "tracklets.jsonl"
    tracklets.write_text(
        json.dumps(
            {
                "meta": {"seq": "Clip_001", "track_id": "t1", "label": 1, "dataset_source": "aot"},
                "rows": rows,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return frame_root, tracklets


def _write_vatd_motion_action_smoke_data(tmp_path):
    frame_root = tmp_path / "vatd_frames"
    frame_root.mkdir()
    items = []
    for track_id, label, y1, brightness in [("pos", 1, 20, 255), ("neg", 0, 40, 96)]:
        rows = []
        for frame_id in range(6):
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            x1 = 10 + frame_id if label else 35
            cv2.rectangle(image, (x1, y1), (x1 + 5, y1 + 5), (brightness, brightness, brightness), -1)
            cv2.imwrite(str(frame_root / f"Clip_001_{track_id}_{frame_id:05d}.png"), image)
            rows.append(
                {
                    "seq": f"Clip_001_{track_id}",
                    "track_id": track_id,
                    "frame_id": frame_id,
                    "bbox": [x1, y1, x1 + 5, y1 + 5],
                    "objectness": 0.8 if label else 0.2,
                    "visible": True,
                }
            )
        items.append({"meta": {"seq": f"Clip_001_{track_id}", "track_id": track_id, "label": label, "dataset_source": "smoke"}, "rows": rows})
    tracklets = tmp_path / "vatd_tracklets.jsonl"
    tracklets.write_text("\n".join(json.dumps(item) for item in items) + "\n", encoding="utf-8")
    return frame_root, tracklets


def test_video_action_tracklet_dataset_loads_crops_and_actions(tmp_path):
    frame_root, tracklets = _write_video_action_smoke_data(tmp_path)

    dataset = VideoActionTrackletDataset(tracklets, frame_root=frame_root, past_len=3, future_len=2, crop_size=32, image_size=(64, 64))
    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["crops"].shape == (3, 3, 32, 32)
    assert sample["state"].shape == (3, 6)
    assert sample["future_actions"].shape == (2, 4)
    assert torch.isfinite(sample["future_actions"]).all()


def test_train_and_score_video_action_policy_smoke(tmp_path):
    frame_root, tracklets = _write_video_action_smoke_data(tmp_path)
    weights = train_video_action_chunk_policy(
        tracklets,
        tmp_path / "video_action.pt",
        frame_root=frame_root,
        past_len=3,
        future_len=2,
        crop_size=32,
        image_size=(64, 64),
        epochs=1,
        batch_size=2,
        d_model=32,
        nhead=4,
        num_layers=1,
    )
    result = score_tracklets_with_video_action_policy(
        tracklets,
        weights,
        tmp_path / "scores.jsonl",
        frame_root=frame_root,
        error_scale=0.1,
    )
    rows = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert weights.exists()
    assert result.summary["tracklets"] == 1
    assert rows[0]["dynamics_score_mode"] == "video_action_chunk_transformer"
    assert 0.0 <= rows[0]["dynamics_score"] <= 1.0


def test_train_and_score_video_action_multihead_policy_smoke(tmp_path):
    frame_root, tracklets = _write_video_action_smoke_data(tmp_path)
    weights = train_video_action_multihead_policy(
        tracklets,
        tmp_path / "video_action_multihead.pt",
        frame_root=frame_root,
        past_len=2,
        future_len=1,
        crop_size=32,
        image_size=(64, 64),
        epochs=1,
        batch_size=2,
        d_model=32,
        nhead=4,
        num_layers=1,
        confidence_target="max",
    )
    result = score_tracklets_with_video_action_multihead_policy(
        tracklets,
        weights,
        tmp_path / "multihead_scores.jsonl",
        frame_root=frame_root,
        error_scale=0.1,
        fusion_mode="dynamics_times_predicted_confidence",
    )
    rows = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert weights.exists()
    assert result.summary["tracklets"] == 1
    assert rows[0]["dynamics_score_mode"] == "video_action_multihead_transformer"
    assert 0.0 <= rows[0]["dynamics_score"] <= 1.0
    assert 0.0 <= rows[0]["predicted_confidence_score"] <= 1.0
    assert 0.0 <= rows[0]["video_action_model_fusion_score"] <= 1.0


def test_train_and_score_vatd_motion_action_policy_smoke(tmp_path):
    frame_root, tracklets = _write_vatd_motion_action_smoke_data(tmp_path)
    weights = train_vatd_motion_action_policy(
        tracklets,
        tmp_path / "vatd_motion_action.pt",
        frame_root=frame_root,
        image_name_template="{seq}_{frame_id_05d}.png",
        past_len=2,
        future_len=1,
        crop_size=32,
        image_size=(64, 64),
        epochs=1,
        batch_size=2,
        d_model=32,
        nhead=4,
        num_layers=1,
        action_loss_weight=0.2,
    )
    result = score_tracklets_with_vatd_motion_action_policy(
        tracklets,
        weights,
        tmp_path / "vatd_scores.jsonl",
        frame_root=frame_root,
        image_name_template="{seq}_{frame_id_05d}.png",
        error_scale=0.1,
        fusion_mode="motion_times_action_consistency",
    )
    rows = [json.loads(line) for line in result.out_path.read_text(encoding="utf-8").splitlines()]

    assert weights.exists()
    assert result.summary["tracklets"] == 2
    assert {row["dynamics_score_mode"] for row in rows} == {"vatd_motion_action_transformer"}
    assert all(0.0 <= row["motion_action_score"] <= 1.0 for row in rows)
    assert all(0.0 <= row["vatd_score"] <= 1.0 for row in rows)

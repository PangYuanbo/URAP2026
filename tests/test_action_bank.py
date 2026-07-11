from __future__ import annotations

import numpy as np
import torch

from qstr_dronedet.tracking.action_bank import (
    ActionBankConfig,
    DualTimeActionBankTransformer,
    action_token,
    build_action_bank,
    row_timestamp,
    score_candidate,
)


def _row(frame_id: int, x: float, *, fps: float, camera_dx: float = 0.0, score: float = 0.8) -> dict:
    return {
        "frame_id": frame_id,
        "fps": fps,
        "bbox": [x, 40.0, x + 10.0, 50.0],
        "image_width": 100.0,
        "image_height": 100.0,
        "camera_dx": camera_dx,
        "camera_dy": 0.0,
        "score": score,
        "visible": True,
    }


def test_timestamp_uses_real_fps_and_explicit_time() -> None:
    assert np.isclose(row_timestamp({"frame_id": 75, "fps": 75}), 1.0)
    assert np.isclose(row_timestamp({"frame_id": 75, "fps": 25}), 3.0)
    assert np.isclose(row_timestamp({"frame_id": 999, "timestamp_sec": 2.5}), 2.5)


def test_camera_motion_is_removed_before_velocity() -> None:
    config = ActionBankConfig(fps_fallback=10.0)
    previous = _row(0, 10.0, fps=10.0)
    current = _row(1, 12.0, fps=10.0, camera_dx=2.0)
    token = action_token(previous, current, config)
    assert np.isclose(token.values[1], 0.0, atol=1e-6)
    assert np.isclose(token.values[3], 0.0, atol=1e-6)
    assert np.isclose(token.values[16], 0.2, atol=1e-6)


def test_bank_windows_follow_seconds_not_frame_count() -> None:
    config = ActionBankConfig(short_tokens=10, long_tokens=15, fps_fallback=25.0)
    rows = [_row(frame, 10.0 + frame * 0.2, fps=25.0) for frame in range(101)]
    snapshot = build_action_bank(rows, config=config)
    assert snapshot.short_tokens.shape == (10, 18)
    assert snapshot.long_tokens.shape == (15, 18)
    assert snapshot.short_mask.sum() == 10
    assert snapshot.long_mask.sum() > 10
    valid_short_ages = snapshot.short_tokens[snapshot.short_mask > 0, 0]
    assert valid_short_ages.min() >= 0.0
    assert valid_short_ages.max() <= 1.0


def test_candidate_score_prefers_motion_continuation() -> None:
    config = ActionBankConfig(short_tokens=8, long_tokens=12, fps_fallback=10.0)
    rows = [_row(frame, 10.0 + frame, fps=10.0) for frame in range(31)]
    snapshot = build_action_bank(rows, config=config)
    previous = rows[-1]
    good = _row(31, 41.0, fps=10.0, score=0.5)
    bad = _row(31, 10.0, fps=10.0, score=0.9)
    assert score_candidate(previous, good, snapshot).score > score_candidate(previous, bad, snapshot).score


def test_dual_bank_model_runs_on_gpu_or_cpu() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualTimeActionBankTransformer(short_tokens=8, long_tokens=12, future_steps=2).to(device)
    short = torch.randn(3, 8, 18, device=device)
    long = torch.randn(3, 12, 18, device=device)
    short_mask = torch.ones(3, 8, device=device)
    long_mask = torch.ones(3, 12, device=device)
    motion, future, reliability = model(short, short_mask, long, long_mask)
    assert motion.shape == (3,)
    assert future.shape == (3, 2, 4)
    assert reliability.shape == (3,)
    assert motion.device.type == device.type



def test_online_tracker_reidentifies_after_short_gap() -> None:
    from qstr_dronedet.tracking.action_bank import OnlineActionBankTracker

    tracker = OnlineActionBankTracker(ActionBankConfig(fps_fallback=10.0), match_threshold=0.2)
    track_id = None
    for frame in range(5):
        result = tracker.update([_row(frame, 10.0 + frame, fps=10.0)])
        track_id = result[0]["action_bank_track_id"]
    result = tracker.update([_row(15, 25.0, fps=10.0)])
    assert result[0]["action_bank_track_id"] == track_id
    assert result[0]["action_bank_reidentified"] is True


def test_online_tracker_expires_after_three_seconds() -> None:
    from qstr_dronedet.tracking.action_bank import OnlineActionBankTracker

    tracker = OnlineActionBankTracker(ActionBankConfig(fps_fallback=10.0), match_threshold=0.2, max_dormant_seconds=3.0)
    first_id = tracker.update([_row(0, 10.0, fps=10.0)])[0]["action_bank_track_id"]
    second_id = tracker.update([_row(40, 50.0, fps=10.0)])[0]["action_bank_track_id"]
    assert second_id != first_id


def test_zero_shot_evaluator_tracks_generic_motion() -> None:
    from tools.evaluate_action_bank_zeroshot import evaluate_sequence, summarize

    frames = []
    for frame in range(20):
        x = 5.0 + frame
        frames.append({
            "timestamp_sec": frame / 10.0,
            "gt_bbox": [x, 5.0, x + 10.0, 15.0],
            "candidates": [
                {"bbox": [x, 5.0, x + 10.0, 15.0], "score": 0.6, "image_width": 100, "image_height": 100},
                {"bbox": [80.0 - frame, 60.0, 90.0 - frame, 70.0], "score": 0.9, "image_width": 100, "image_height": 100},
            ],
        })
    result = evaluate_sequence({"seq": "generic-object", "frames": frames}, ActionBankConfig(fps_fallback=10.0))
    summary = summarize([result], 70.0)
    assert summary["success_auc_percent"] >= 70.0
    assert summary["target_met"] is True


def test_sequence_specific_fps_controls_real_time() -> None:
    config = ActionBankConfig(fps_fallback=25.0, sequence_fps={"fast": 50.0})
    previous = {**_row(0, 10.0, fps=25.0), "seq": "fast"}
    current = {**_row(5, 15.0, fps=25.0), "seq": "fast"}
    previous.pop("fps")
    current.pop("fps")
    token = action_token(previous, current, config)
    assert np.isclose(token.dt, 0.1)
    assert np.isclose(token.values[3], 0.5)

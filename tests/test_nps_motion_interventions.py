from __future__ import annotations

import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from qstr_dronedet.nps_motion_interventions import (
    FlowQuality,
    build_clip,
    interpolate_boxes,
    make_time_map,
    output_paths,
    validate_intervention,
    write_dataset_metadata,
)
from tools.summarize_nps_motion_robustness import bootstrap_drop, evaluate_frames


class FakeInterpolator:
    def interpolate(self, left: np.ndarray, right: np.ndarray, alpha: float) -> tuple[np.ndarray, FlowQuality]:
        return cv2.addWeighted(left, 1.0 - alpha, right, alpha, 0.0), FlowQuality(0.0, 0.0, True)


def make_source(root: Path, count: int = 4) -> tuple[Path, Path]:
    frames = root / "AllFrames" / "test"
    labels = root / "NPSvisdroneStyle" / "test" / "labels"
    frames.mkdir(parents=True)
    labels.mkdir(parents=True)
    for frame_id in range(1, count + 1):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.rectangle(image, (8 + frame_id * 3, 16), (14 + frame_id * 3, 22), (255, 255, 255), -1)
        assert cv2.imwrite(str(frames / f"Clip_41_{frame_id:05d}.png"), image)
        (labels / f"Clip_41_{frame_id - 1:05d}.txt").write_text(
            f"0 {(11 + frame_id * 3) / 64:.8f} {19 / 48:.8f} {6 / 64:.8f} {6 / 48:.8f}\n",
            encoding="utf-8",
        )
    return frames, labels


def test_time_maps_cover_endpoints_and_are_monotonic() -> None:
    expected_lengths = {"original": 5, "slow_0p5": 9, "fast_2x": 3, "accelerate_g2": 5, "decelerate_g2": 5}
    for intervention, expected_length in expected_lengths.items():
        mapping = make_time_map(intervention, 5)
        assert len(mapping) == expected_length
        assert mapping[0] == 0
        assert mapping[-1] == 4
        assert np.all(np.diff(mapping) > 0)


def test_interpolate_boxes_matches_reordered_targets() -> None:
    left = [[0, 0.2, 0.3, 0.1, 0.1], [0, 0.8, 0.7, 0.08, 0.08]]
    right = [[0, 0.75, 0.7, 0.08, 0.08], [0, 0.3, 0.3, 0.1, 0.1]]
    result = interpolate_boxes(left, right, 0.5)
    assert result.valid
    assert [round(box[1], 3) for box in result.boxes] == [0.25, 0.775]


def test_interpolate_boxes_falls_back_when_target_count_changes() -> None:
    result = interpolate_boxes([[0, 0.2, 0.3, 0.1, 0.1]], [], 0.5)
    assert not result.valid
    assert result.reason == "target_count_changed"


def test_build_clip_writes_both_dataset_formats(tmp_path: Path) -> None:
    frames, labels = make_source(tmp_path / "source")
    intervention_root = tmp_path / "out" / "slow_0p5"
    summary = build_clip(frames, labels, intervention_root, "test", "Clip_41", "slow_0p5", FakeInterpolator())
    assert summary["source_frames"] == 4
    assert summary["output_frames"] == 7
    assert summary["fallback_frames"] == 0
    for frame_id in range(1, 8):
        paths = output_paths(intervention_root, "test", "Clip_41", frame_id)
        assert all(path.exists() for path in paths.values())
    first_motion = cv2.imread(str(output_paths(intervention_root, "test", "Clip_41", 1)["yolomg_motion"]), cv2.IMREAD_GRAYSCALE)
    second_motion = cv2.imread(str(output_paths(intervention_root, "test", "Clip_41", 2)["yolomg_motion"]), cv2.IMREAD_GRAYSCALE)
    assert not np.any(first_motion)
    assert np.any(second_motion)
    records = [json.loads(line) for line in (intervention_root / "manifests" / "test" / "Clip_41.jsonl").read_text().splitlines()]
    assert [record["output_frame_id"] for record in records] == list(range(1, 8))
    assert records[1]["synthetic"]
    write_dataset_metadata(intervention_root, "slow_0p5", {"test": {41: 7}})
    with (intervention_root / "TransVisDrone" / "Videos" / "test" / "video_length_dict.pkl").open("rb") as handle:
        assert pickle.load(handle) == {41: 7}
    integrity = validate_intervention(intervention_root, "slow_0p5", ["test"])
    assert integrity["valid"]
    assert integrity["total_frames"] == integrity["total_labels"] == 7


def test_invalid_flow_uses_nearest_anchor(tmp_path: Path) -> None:
    class InvalidInterpolator:
        def interpolate(self, left: np.ndarray, right: np.ndarray, alpha: float) -> tuple[np.ndarray, FlowQuality]:
            return left.copy(), FlowQuality(9.0, 1.0, False)

    frames, labels = make_source(tmp_path / "source", count=2)
    root = tmp_path / "out" / "slow_0p5"
    summary = build_clip(frames, labels, root, "test", "Clip_41", "slow_0p5", InvalidInterpolator())
    assert summary["fallback_frames"] == 1
    record = json.loads((root / "manifests" / "test" / "Clip_41.jsonl").read_text().splitlines()[1])
    assert record["fallback_reason"] == "flow_inconsistent"
    assert record["label_mode"] == "nearest_anchor_fallback"


def test_unified_metrics_and_bootstrap_detect_drop() -> None:
    ground_truth = [0, 0.5, 0.5, 0.2, 0.2]
    perfect_prediction = [0, 0.5, 0.5, 0.2, 0.2, 0.9]
    perfect = evaluate_frames([{"gt": [ground_truth], "pred": [perfect_prediction]}])
    missed = evaluate_frames([{"gt": [ground_truth], "pred": []}])
    assert perfect["tp"] == 1 and perfect["fp"] == 0 and perfect["fn"] == 0
    assert perfect["map50"] > 0.99
    assert missed["recall"] == 0.0
    point, lower, upper = bootstrap_drop({"Clip_41": 1.0, "Clip_42": 1.0}, {"Clip_41": 0.5, "Clip_42": 0.5}, 100, 7)
    assert point == lower == upper == 0.5

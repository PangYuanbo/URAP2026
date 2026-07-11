from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("build_ard100_short_tracklets", TOOLS / "build_ard100_short_tracklets.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def box(x: float) -> np.ndarray:
    return np.asarray((x, 10.0, 8.0, 6.0), dtype=np.float32)


def test_matches_nps_gap_rule_and_length_cap() -> None:
    selected = {frame: box(frame) for frame in range(1, 21)}
    selected.update({24: box(24), 25: box(25), 26: box(26), 30: box(30)})
    items = MODULE.split_tracklets(9, selected, max_gap=2, max_frames=8, min_visible_frames=3, min_visibility=0.5)
    assert [item.frame_ids for item in items] == [tuple(range(1, 9)), tuple(range(9, 17)), tuple(range(17, 21)), (24, 25, 26)]


def test_preserves_short_occlusion_inside_tracklet() -> None:
    selected = {1: box(1), 2: box(2), 4: box(4), 5: box(5)}
    items = MODULE.split_tracklets(3, selected, max_gap=2, max_frames=166, min_visible_frames=4, min_visibility=0.5)
    assert len(items) == 1
    assert items[0].frame_ids == (1, 2, 3, 4, 5)
    assert items[0].boxes[2] is None


def test_rejects_low_visibility() -> None:
    selected = {1: box(1), 4: box(4), 7: box(7), 10: box(10)}
    items = MODULE.split_tracklets(5, selected, max_gap=2, max_frames=10, min_visible_frames=4, min_visibility=0.5)
    assert items == []

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image

TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "materialize_ard100_short166_local",
    TOOLS / "materialize_ard100_short166_local.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_rebuilds_a_decodable_rectangular_mask(tmp_path: Path) -> None:
    target = tmp_path / "mask.png"
    target.write_bytes(b"truncated png")
    MODULE.write_box_mask(target, "2,3,4,5", (12, 10))
    MODULE.verify_image(target, (12, 10), decode=True)
    with Image.open(target) as mask:
        assert mask.getpixel((2, 3)) == 1
        assert mask.getpixel((5, 7)) == 1
        assert mask.getpixel((6, 7)) == 0


def test_zero_box_creates_empty_mask(tmp_path: Path) -> None:
    target = tmp_path / "empty.png"
    MODULE.write_box_mask(target, "0,0,0,0", (12, 10))
    with Image.open(target) as mask:
        assert mask.getbbox() is None

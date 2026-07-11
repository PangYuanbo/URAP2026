from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAM2_ROOT = ROOT / "third_party" / "samurai" / "sam2"
sys.path.insert(0, str(SAM2_ROOT))

from training.dataset.vos_raw_dataset import WindowedPNGRawDataset


def make_dataset(tmp_path: Path, img_subdir: str | None) -> WindowedPNGRawDataset:
    sequence = "clip_001"
    image_root = tmp_path / "images" / sequence
    if img_subdir:
        image_root /= img_subdir
    mask_root = tmp_path / "masks" / sequence
    visibility_root = tmp_path / "tracks" / sequence
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    visibility_root.mkdir(parents=True)
    for frame_id in range(1, 5):
        Image.new("RGB", (8, 8), color=(frame_id, 0, 0)).save(
            image_root / f"{frame_id:08d}.jpg"
        )
        Image.new("P", (8, 8), color=1).save(mask_root / f"{frame_id:08d}.png")
    (visibility_root / "full_occlusion.txt").write_text("0,0,0,0", encoding="ascii")
    file_list = tmp_path / "train.txt"
    file_list.write_text(sequence + "\n", encoding="ascii")
    return WindowedPNGRawDataset(
        img_folder=str(tmp_path / "images"),
        gt_folder=str(tmp_path / "masks"),
        file_list_txt=str(file_list),
        visibility_folder=str(tmp_path / "tracks"),
        window_size=2,
        window_stride=2,
        img_subdir=img_subdir,
    )


def test_standard_layout_uses_sequence_root(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, img_subdir=None)
    assert len(dataset) == 2
    video, _ = dataset.get_video(0)
    assert [Path(frame.image_path).name for frame in video.frames] == [
        "00000001.jpg",
        "00000002.jpg",
    ]


def test_local_layout_uses_img_subdirectory(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, img_subdir="img")
    assert len(dataset) == 2
    video, _ = dataset.get_video(1)
    assert [Path(frame.image_path).name for frame in video.frames] == [
        "00000003.jpg",
        "00000004.jpg",
    ]

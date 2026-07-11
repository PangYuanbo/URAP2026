from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qstr_dronedet.native_video_detector import NPSClipDataset


def _write_synthetic_nps(root: Path, frames: int = 6) -> tuple[Path, Path]:
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True)
    gt_csv = root / "gt.csv"
    for frame_id in range(1, frames + 1):
        img = Image.new("RGB", (96, 64), color=(30, frame_id * 30 % 255, 70))
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


def test_native_video_frame_cache_roundtrip(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    frames_dir, gt_csv = _write_synthetic_nps(tmp_path / "data")
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "run"
    pkl_path = tmp_path / "predictionsgt.pkl"

    subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "build_native_video_frame_cache.py"),
            "--frames-dir",
            str(frames_dir),
            "--cache-dir",
            str(cache_dir),
            "--image-size",
            "64",
            "--log-every",
            "2",
        ],
        cwd=repo,
        check=True,
    )
    assert (cache_dir / "cache_summary.json").exists()
    assert len(list(cache_dir.glob("Clip_1_*.pt"))) == 6
    assert not list(cache_dir.glob("*.tmp.*"))

    no_cache = NPSClipDataset(frames_dir, gt_csv, clip_len=4, future_len=2, image_size=64)
    with_cache = NPSClipDataset(frames_dir, gt_csv, clip_len=4, future_len=2, image_size=64, cache_dir=cache_dir)
    assert torch.equal(no_cache[0]["image_size"], with_cache[0]["image_size"])
    assert torch.allclose(no_cache[0]["clip"], with_cache[0]["clip"])
    assert len(no_cache[0]["future_boxes"]) == 3

    subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "train_native_video_detector.py"),
            "--frames-dir",
            str(frames_dir),
            "--gt-csv",
            str(gt_csv),
            "--cache-dir",
            str(cache_dir),
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
            "--future-len",
            "2",
            "--clip-len",
            "4",
            "--num-workers",
            "0",
            "--channels-last",
            "--tf32",
            "--cudnn-benchmark",
            "--device",
            "cpu",
        ],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "export_native_video_predictionsgt.py"),
            "--weights",
            str(out_dir / "native_video_detector.pt"),
            "--frames-dir",
            str(frames_dir),
            "--gt-csv",
            str(gt_csv),
            "--cache-dir",
            str(cache_dir),
            "--out-pkl",
            str(pkl_path),
            "--batch-size",
            "2",
            "--top-k",
            "4",
            "--device",
            "cpu",
        ],
        cwd=repo,
        check=True,
    )
    assert pkl_path.exists()


def test_native_video_dataset_hflip_augments_clip_and_boxes(tmp_path: Path) -> None:
    frames_dir, gt_csv = _write_synthetic_nps(tmp_path / "data")
    plain = NPSClipDataset(frames_dir, gt_csv, clip_len=4, future_len=2, image_size=64)
    flipped = NPSClipDataset(frames_dir, gt_csv, clip_len=4, future_len=2, image_size=64, augment_hflip_prob=1.0)

    plain_item = plain[0]
    flipped_item = flipped[0]
    assert torch.allclose(torch.flip(plain_item["clip"], dims=[3]), flipped_item["clip"])
    assert torch.allclose(flipped_item["boxes"][:, 0], 1.0 - plain_item["boxes"][:, 0])
    assert torch.allclose(flipped_item["boxes"][:, 1:], plain_item["boxes"][:, 1:])
    assert len(flipped_item["future_boxes"]) == 3
    assert torch.allclose(flipped_item["future_boxes"][1][:, 0], 1.0 - plain_item["future_boxes"][1][:, 0])
    assert torch.allclose(flipped_item["future_boxes"][1][:, 1:], plain_item["future_boxes"][1][:, 1:])


def test_native_video_dataset_ignores_wrong_size_cache(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    frames_dir, gt_csv = _write_synthetic_nps(tmp_path / "data")
    cache_dir = tmp_path / "cache32"
    subprocess.run(
        [
            sys.executable,
            str(repo / "tools" / "build_native_video_frame_cache.py"),
            "--frames-dir",
            str(frames_dir),
            "--cache-dir",
            str(cache_dir),
            "--image-size",
            "32",
        ],
        cwd=repo,
        check=True,
    )

    no_cache = NPSClipDataset(frames_dir, gt_csv, clip_len=4, future_len=2, image_size=64)
    wrong_cache = NPSClipDataset(frames_dir, gt_csv, clip_len=4, future_len=2, image_size=64, cache_dir=cache_dir)

    assert wrong_cache[0]["clip"].shape == (4, 3, 64, 64)
    assert torch.allclose(no_cache[0]["clip"], wrong_cache[0]["clip"])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="native_video_cache_") as tmp:
        test_native_video_frame_cache_roundtrip(Path(tmp))
    with tempfile.TemporaryDirectory(prefix="native_video_hflip_") as tmp:
        test_native_video_dataset_hflip_augments_clip_and_boxes(Path(tmp))
    with tempfile.TemporaryDirectory(prefix="native_video_wrong_cache_") as tmp:
        test_native_video_dataset_ignores_wrong_size_cache(Path(tmp))

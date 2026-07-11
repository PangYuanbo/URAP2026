from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def _write_gt(path: Path, frame_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["seq,frame_id,x1,y1,x2,y2,video_path"]
    for idx, frame_name in enumerate(frame_names, start=1):
        lines.append(f"Clip_1,{idx},1,2,3,4,Clip_1/{frame_name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _touch_pngs(split_dir: Path, count: int) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(1, count + 1):
        (split_dir / f"Clip_1_{idx:05d}.png").write_bytes(b"")


def _run_ready(repo: Path, data_root: Path, min_train: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(repo / "tools" / "check_native_video_dataset_ready.ps1"),
            "-DataRoot",
            str(data_root),
            "-MinTrainFrames",
            str(min_train),
            "-UseGtExpectedFrames",
            "0",
            "-VerifyGtFrameFiles",
            "0",
        ],
        cwd=repo,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_native_video_dataset_ready_gate() -> None:
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="native_video_ready_") as tmp:
        data_root = Path(tmp) / "URAP_datasets"
        allframes = data_root / "TransVisDrone" / "NPS" / "AllFrames"

        _touch_pngs(allframes / "train", 4)
        _touch_pngs(allframes / "val", 2)
        _touch_pngs(allframes / "test", 2)
        not_ready = _run_ready(repo, data_root, min_train=5)
        assert not_ready.returncode != 0
        assert "Status: NOT READY" in not_ready.stdout, not_ready.stdout + not_ready.stderr

        _touch_pngs(allframes / "train", 5)
        ready = _run_ready(repo, data_root, min_train=5)
        assert ready.returncode == 0, ready.stdout + ready.stderr
        assert "Status: READY" in ready.stdout


def test_native_video_dataset_ready_rejects_missing_gt_referenced_frames() -> None:
    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="native_video_ready_gt_") as tmp:
        root = Path(tmp)
        data_root = root / "URAP_datasets"
        allframes = data_root / "TransVisDrone" / "NPS" / "AllFrames"
        gt_train = root / "gt_train.csv"
        gt_val = root / "gt_val.csv"
        gt_test = root / "gt_test.csv"

        _touch_pngs(allframes / "train", 2)
        _touch_pngs(allframes / "val", 1)
        _touch_pngs(allframes / "test", 1)
        _write_gt(gt_train, ["Clip_1_00001.png", "Clip_1_00003.png"])
        _write_gt(gt_val, ["Clip_1_00001.png"])
        _write_gt(gt_test, ["Clip_1_00001.png"])

        missing = subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(repo / "tools" / "check_native_video_dataset_ready.ps1"),
                "-DataRoot",
                str(data_root),
                "-MinTrainFrames",
                "1",
                "-MinValFrames",
                "1",
                "-MinTestFrames",
                "1",
                "-TrainGtCsv",
                str(gt_train),
                "-ValGtCsv",
                str(gt_val),
                "-TestGtCsv",
                str(gt_test),
            ],
            cwd=repo,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert missing.returncode != 0
        assert "Status: NOT READY" in missing.stdout, missing.stdout + missing.stderr
        assert "missing_gt_frames=1" in missing.stdout, missing.stdout
        assert "Clip_1_00003.png" in missing.stdout, missing.stdout

        (allframes / "train" / "Clip_1_00003.png").write_bytes(b"")
        ready = subprocess.run(
            [
                "powershell",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(repo / "tools" / "check_native_video_dataset_ready.ps1"),
                "-DataRoot",
                str(data_root),
                "-MinTrainFrames",
                "1",
                "-MinValFrames",
                "1",
                "-MinTestFrames",
                "1",
                "-TrainGtCsv",
                str(gt_train),
                "-ValGtCsv",
                str(gt_val),
                "-TestGtCsv",
                str(gt_test),
            ],
            cwd=repo,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert ready.returncode == 0, ready.stdout + ready.stderr
        assert "missing_gt_frames=0" in ready.stdout
        assert "Status: READY" in ready.stdout


if __name__ == "__main__":
    test_native_video_dataset_ready_gate()
    test_native_video_dataset_ready_rejects_missing_gt_referenced_frames()
